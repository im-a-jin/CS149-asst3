#include <string>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/set_operations.h>
#include <thrust/remove.h>
#include <thrust/adjacent_difference.h>
#include <thrust/gather.h>

// #include <thrust/mr/universal_memory_resource.h>
// #include <thrust/mr/disjoint_pool.h>
// #include <thrust/mr/allocator.h>



#include "cudaRenderer.h"
#include "image.h"
#include "noise.h"
#include "sceneLoader.h"
#include "util.h"

////////////////////////////////////////////////////////////////////////////////////////
// Putting all the cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

struct GlobalConstants {

    SceneName sceneName;

    int numCircles;
    float* position;
    float* velocity;
    float* color;
    float* radius;

    int imageWidth;
    int imageHeight;
    float* imageData;
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuConstRendererParams;

// read-only lookup tables used to quickly compute noise (needed by
// advanceAnimation for the snowflake scene)
__constant__ int    cuConstNoiseYPermutationTable[256];
__constant__ int    cuConstNoiseXPermutationTable[256];
__constant__ float  cuConstNoise1DValueTable[256];

// color ramp table needed for the color ramp lookup shader
#define COLOR_MAP_SIZE 5
__constant__ float  cuConstColorRamp[COLOR_MAP_SIZE][3];

// block dimensions
#define BLOCK_DIM_X 32
#define BLOCK_DIM_Y 32
#define SCAN_BLOCK_DIM (BLOCK_DIM_X*BLOCK_DIM_Y)
#define BUFFER_SIZE 256

// including parts of the CUDA code from external files to keep this
// file simpler and to seperate code that should not be modified
#include "noiseCuda.cu_inl"
#include "lookupColor.cu_inl"
#include "circleBoxTest.cu_inl"
#include "exclusiveScan.cu_inl"


// kernelClearImageSnowflake -- (CUDA device code)
//
// Clear the image, setting the image to the white-gray gradation that
// is used in the snowflake image
__global__ void kernelClearImageSnowflake() {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float shade = .4f + .45f * static_cast<float>(height-imageY) / height;
    float4 value = make_float4(shade, shade, shade, 1.f);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelClearImage --  (CUDA device code)
//
// Clear the image, setting all pixels to the specified color rgba
__global__ void kernelClearImage(float r, float g, float b, float a) {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float4 value = make_float4(r, g, b, a);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelAdvanceFireWorks
// 
// Update the position of the fireworks (if circle is firework)
__global__ void kernelAdvanceFireWorks() {
    const float dt = 1.f / 60.f;
    const float pi = 3.14159;
    const float maxDist = 0.25f;

    float* velocity = cuConstRendererParams.velocity;
    float* position = cuConstRendererParams.position;
    float* radius = cuConstRendererParams.radius;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles)
        return;

    if (0 <= index && index < NUM_FIREWORKS) { // firework center; no update 
        return;
    }

    // determine the fire-work center/spark indices
    int fIdx = (index - NUM_FIREWORKS) / NUM_SPARKS;
    int sfIdx = (index - NUM_FIREWORKS) % NUM_SPARKS;

    int index3i = 3 * fIdx;
    int sIdx = NUM_FIREWORKS + fIdx * NUM_SPARKS + sfIdx;
    int index3j = 3 * sIdx;

    float cx = position[index3i];
    float cy = position[index3i+1];

    // update position
    position[index3j] += velocity[index3j] * dt;
    position[index3j+1] += velocity[index3j+1] * dt;

    // fire-work sparks
    float sx = position[index3j];
    float sy = position[index3j+1];

    // compute vector from firework-spark
    float cxsx = sx - cx;
    float cysy = sy - cy;

    // compute distance from fire-work 
    float dist = sqrt(cxsx * cxsx + cysy * cysy);
    if (dist > maxDist) { // restore to starting position 
        // random starting position on fire-work's rim
        float angle = (sfIdx * 2 * pi)/NUM_SPARKS;
        float sinA = sin(angle);
        float cosA = cos(angle);
        float x = cosA * radius[fIdx];
        float y = sinA * radius[fIdx];

        position[index3j] = position[index3i] + x;
        position[index3j+1] = position[index3i+1] + y;
        position[index3j+2] = 0.0f;

        // travel scaled unit length 
        velocity[index3j] = cosA/5.0;
        velocity[index3j+1] = sinA/5.0;
        velocity[index3j+2] = 0.0f;
    }
}

// kernelAdvanceHypnosis   
//
// Update the radius/color of the circles
__global__ void kernelAdvanceHypnosis() { 
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles) 
        return; 

    float* radius = cuConstRendererParams.radius; 

    float cutOff = 0.5f;
    // place circle back in center after reaching threshold radisus 
    if (radius[index] > cutOff) { 
        radius[index] = 0.02f; 
    } else { 
        radius[index] += 0.01f; 
    }   
}   


// kernelAdvanceBouncingBalls
// 
// Update the position of the balls
__global__ void kernelAdvanceBouncingBalls() { 
    const float dt = 1.f / 60.f;
    const float kGravity = -2.8f; // sorry Newton
    const float kDragCoeff = -0.8f;
    const float epsilon = 0.001f;

    int index = blockIdx.x * blockDim.x + threadIdx.x; 
   
    if (index >= cuConstRendererParams.numCircles) 
        return; 

    float* velocity = cuConstRendererParams.velocity; 
    float* position = cuConstRendererParams.position; 

    int index3 = 3 * index;
    // reverse velocity if center position < 0
    float oldVelocity = velocity[index3+1];
    float oldPosition = position[index3+1];

    if (oldVelocity == 0.f && oldPosition == 0.f) { // stop-condition 
        return;
    }

    if (position[index3+1] < 0 && oldVelocity < 0.f) { // bounce ball 
        velocity[index3+1] *= kDragCoeff;
    }

    // update velocity: v = u + at (only along y-axis)
    velocity[index3+1] += kGravity * dt;

    // update positions (only along y-axis)
    position[index3+1] += velocity[index3+1] * dt;

    if (fabsf(velocity[index3+1] - oldVelocity) < epsilon
        && oldPosition < 0.0f
        && fabsf(position[index3+1]-oldPosition) < epsilon) { // stop ball 
        velocity[index3+1] = 0.f;
        position[index3+1] = 0.f;
    }
}

// kernelAdvanceSnowflake -- (CUDA device code)
//
// move the snowflake animation forward one time step.  Updates circle
// positions and velocities.  Note how the position of the snowflake
// is reset if it moves off the left, right, or bottom of the screen.
__global__ void kernelAdvanceSnowflake() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    const float dt = 1.f / 60.f;
    const float kGravity = -1.8f; // sorry Newton
    const float kDragCoeff = 2.f;

    int index3 = 3 * index;

    float* positionPtr = &cuConstRendererParams.position[index3];
    float* velocityPtr = &cuConstRendererParams.velocity[index3];

    // loads from global memory
    float3 position = *((float3*)positionPtr);
    float3 velocity = *((float3*)velocityPtr);

    // hack to make farther circles move more slowly, giving the
    // illusion of parallax
    float forceScaling = fmin(fmax(1.f - position.z, .1f), 1.f); // clamp

    // add some noise to the motion to make the snow flutter
    float3 noiseInput;
    noiseInput.x = 10.f * position.x;
    noiseInput.y = 10.f * position.y;
    noiseInput.z = 255.f * position.z;
    float2 noiseForce = cudaVec2CellNoise(noiseInput, index);
    noiseForce.x *= 7.5f;
    noiseForce.y *= 5.f;

    // drag
    float2 dragForce;
    dragForce.x = -1.f * kDragCoeff * velocity.x;
    dragForce.y = -1.f * kDragCoeff * velocity.y;

    // update positions
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;

    // update velocities
    velocity.x += forceScaling * (noiseForce.x + dragForce.y) * dt;
    velocity.y += forceScaling * (kGravity + noiseForce.y + dragForce.y) * dt;

    float radius = cuConstRendererParams.radius[index];

    // if the snowflake has moved off the left, right or bottom of
    // the screen, place it back at the top and give it a
    // pseudorandom x position and velocity.
    if ( (position.y + radius < 0.f) ||
         (position.x + radius) < -0.f ||
         (position.x - radius) > 1.f)
    {
        noiseInput.x = 255.f * position.x;
        noiseInput.y = 255.f * position.y;
        noiseInput.z = 255.f * position.z;
        noiseForce = cudaVec2CellNoise(noiseInput, index);

        position.x = .5f + .5f * noiseForce.x;
        position.y = 1.35f + radius;

        // restart from 0 vertical velocity.  Choose a
        // pseudo-random horizontal velocity.
        velocity.x = 2.f * noiseForce.y;
        velocity.y = 0.f;
    }

    // store updated positions and velocities to global memory
    *((float3*)positionPtr) = position;
    *((float3*)velocityPtr) = velocity;
}

// shadePixel -- (CUDA device code)
//
// given a pixel and a circle, determines the contribution to the
// pixel from the circle.  Update of the image is done in this
// function.  Called by kernelRenderCircles()
__device__ __inline__ void
shadePixelSnowflake(int circleIndex, float2 pixelCenter, float3 p, float4* imagePtr) {

    float diffX = p.x - pixelCenter.x;
    float diffY = p.y - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;

    float rad = cuConstRendererParams.radius[circleIndex];;
    float maxDist = rad * rad;

    // circle does not contribute to the image
    if (pixelDist > maxDist)
        return;

    float3 rgb;
    float alpha;

    const float kCircleMaxAlpha = .5f;
    const float falloffScale = 4.f;

    float normPixelDist = sqrt(pixelDist) / rad;
    rgb = lookupColor(normPixelDist);

    float maxAlpha = .6f + .4f * (1.f-p.z);
    maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
    alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);

    float oneMinusAlpha = 1.f - alpha;

    // BEGIN SHOULD-BE-ATOMIC REGION
    // global memory read

    float4 existingColor = *imagePtr;
    float4 newColor;
    newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
    newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
    newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
    newColor.w = alpha + existingColor.w;

    // global memory write
    *imagePtr = newColor;

    // END SHOULD-BE-ATOMIC REGION
}

__device__ __inline__ void
shadePixel(int circleIndex, float2 pixelCenter, float3 p, float4* imagePtr) {

    float diffX = p.x - pixelCenter.x;
    float diffY = p.y - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;

    float rad = cuConstRendererParams.radius[circleIndex];;
    float maxDist = rad * rad;

    // circle does not contribute to the image
    if (pixelDist > maxDist)
        return;

    float3 rgb;
    float alpha;

    // simple: each circle has an assigned color
    int index3 = 3 * circleIndex;
    rgb = *(float3*)&(cuConstRendererParams.color[index3]);
    alpha = .5f;

    float oneMinusAlpha = 1.f - alpha;

    // BEGIN SHOULD-BE-ATOMIC REGION
    // global memory read

    float4 existingColor = *imagePtr;
    float4 newColor;
    newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
    newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
    newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
    newColor.w = alpha + existingColor.w;

    // global memory write
    *imagePtr = newColor;

    // END SHOULD-BE-ATOMIC REGION
}

// kernelRenderCircles -- (CUDA device code)
//
// Each kernel is responsible for rendering a single patch.
// Patch dimensions are BLOCK_DIM_X x BLOCK_DIM_Y, and patches
// divide up the entire image width and height evenly.
// Patch 0 is the bottom left corner of the image (x=0,y=0)
// and patches run left to right, bottom to top

__global__ void kernelRenderCircles(
    int* d_mappedPatchIds,
    int* d_mappedCircleIds,
    int* d_indicesToPatches,
    int n_patchesWithWork,
    uint n_mappingsTotal
) {
    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;
    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;


    // Which patch am I responsible for?
    int workerIdx = blockIdx.x;
    int firstMappingIdx = d_indicesToPatches[workerIdx];
    int patchNumber = d_mappedPatchIds[firstMappingIdx];

    // What are the coordinate bounds of this patch?
    int numPatchesPerRow = (imageWidth + BLOCK_DIM_X - 1) / BLOCK_DIM_X;
    int patchRow = patchNumber / numPatchesPerRow;

    int patchXLeft = (patchNumber % numPatchesPerRow) * BLOCK_DIM_X;
    int patchYBottom = patchRow * BLOCK_DIM_Y;

    int patchYTop = min(patchYBottom + BLOCK_DIM_Y, imageHeight);
    int patchXRight = min(patchXLeft + BLOCK_DIM_X, imageWidth);

    // Which pixel am I specifically responsible for?
    int pixelX = patchXLeft + threadIdx.x;
    int pixelY = patchYBottom + threadIdx.y;

    // How many circles are in this patch?
    int nCirclesInPatch;
    if (workerIdx == n_patchesWithWork - 1) {
        // If we're the last worker, count backwards from total mappings
        nCirclesInPatch = n_mappingsTotal - firstMappingIdx;
    } else {
        // Otherwise, check the delta to the next worker's first mapping
        int nextWorkerFirstMappingIdx = d_indicesToPatches[workerIdx + 1];
        nCirclesInPatch = nextWorkerFirstMappingIdx - firstMappingIdx;
    }  

    float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + pixelX)]);
    float4 accum = *imgPtr;
    float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                         invHeight * (static_cast<float>(pixelY) + 0.5f));

    // Render all circles in this patch
    if (cuConstRendererParams.sceneName == SNOWFLAKES || cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {
        for (int i = 0; i < nCirclesInPatch; i++) {
            int circleIdx = d_mappedCircleIds[firstMappingIdx + i];
            float3 p = *(float3*)(&cuConstRendererParams.position[3*circleIdx]);
            shadePixelSnowflake(circleIdx, pixelCenterNorm, p, &accum);
        }
    } else {
        for (int i = 0; i < nCirclesInPatch; i++) {
            int circleIdx = d_mappedCircleIds[firstMappingIdx + i];
            float3 p = *(float3*)(&cuConstRendererParams.position[3*circleIdx]);
            shadePixel(circleIdx, pixelCenterNorm, p, &accum);
        }
    }
    *imgPtr = accum;


    // uint pixelX = blockMinX + threadIdx.x;
    // uint pixelY = blockMinY + threadIdx.y;
    // float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + pixelX)]);
    // float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
    //                                      invHeight * (static_cast<float>(pixelY) + 0.5f));

    // uint totalCirclesBlock = prefixSumInput[SCAN_BLOCK_DIM-1] + prefixSumOutput[SCAN_BLOCK_DIM-1];
    // float4 accum = *imgPtr;
    // if (cuConstRendererParams.sceneName == SNOWFLAKES || cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {
    //     for (uint i = 0; i < totalCirclesBlock; i++) {
    //         uint index = circleIndexBlock[i];
    //         float3 p = *(float3*)(&cuConstRendererParams.position[3*index]);
    //         shadePixelSnowflake(index, pixelCenterNorm, p, &accum);
    //     }
    // } else {
    //     for (uint i = 0; i < totalCirclesBlock; i++) {
    //         uint index = circleIndexBlock[i];
    //         float3 p = *(float3*)(&cuConstRendererParams.position[3*index]);
    //         shadePixel(index, pixelCenterNorm, p, &accum);
    //     }
    // }
    // *imgPtr = accum;




    // uint linearThreadIndex = threadIdx.y * blockDim.x + threadIdx.x;
                                                                     
    // __shared__ uint prefixSumInput[SCAN_BLOCK_DIM];
    // __shared__ uint prefixSumOutput[SCAN_BLOCK_DIM];
    // __shared__ uint prefixSumScratch[2*SCAN_BLOCK_DIM];
    // __shared__ uint circleIndexBlock[BLOCK_DIM_X*BUFFER_SIZE];

    // short imageWidth = cuConstRendererParams.imageWidth;
    // short imageHeight = cuConstRendererParams.imageHeight;
    // float invWidth = 1.f / imageWidth;
    // float invHeight = 1.f / imageHeight;

    // short blockMinX = static_cast<short>(blockIdx.x * BLOCK_DIM_X);
    // short blockMaxX = static_cast<short>(min(blockMinX + BLOCK_DIM_X, imageWidth));
    // short blockMinY = static_cast<short>(blockIdx.y * BLOCK_DIM_Y);
    // short blockMaxY = static_cast<short>(min(blockMinY + BLOCK_DIM_Y, imageHeight));

    // uint numCirclesBlock = (cuConstRendererParams.numCircles + SCAN_BLOCK_DIM - 1) / SCAN_BLOCK_DIM;
    // uint circleStartIndex = linearThreadIndex * numCirclesBlock;
    // uint circleEndIndex = min(circleStartIndex + numCirclesBlock, cuConstRendererParams.numCircles);


    // MapReduce approach

    // Step 1: Workers act as circle taggers: 
    // Process circles and make (circle index, block index) pairs for each block that the circle touches

    // Step 2: Workers act as patch farmers:
    // Given





    // // Get start time in cycles
    // clock_t start = clock64();

    // uint circleIndexThread[BUFFER_SIZE];
    // uint ci = 0;
    // for (uint i = circleStartIndex; i < circleEndIndex; i++) {
    //     float3 p = *(float3*)(&cuConstRendererParams.position[3*i]);
    //     float  rad = cuConstRendererParams.radius[i];
    //     if (circleInBoxConservative(p.x, p.y, rad, 
    //                                 blockMinX * invWidth, 
    //                                 blockMaxX * invWidth, 
    //                                 blockMaxY * invHeight, 
    //                                 blockMinY * invHeight))
    //         circleIndexThread[ci++] = i;
    // }
    // prefixSumInput[linearThreadIndex] = ci;
    // __syncthreads();

    // // count number of circles in block
    // sharedMemExclusiveScan(linearThreadIndex, prefixSumInput, prefixSumOutput, prefixSumScratch, SCAN_BLOCK_DIM);
    // __syncthreads();


    // // Get end time in cycles
    // clock_t end = clock64();

    // // Only one thread per block should print
    // if (linearThreadIndex == 0) {
    //     // Convert to milliseconds (this is approximate and depends on GPU clock speed)
    //     // Assuming a clock rate of 1.5 GHz
    //     float milliseconds = (end - start) / 1500.0f;
    //     // // // printf("Block [%d,%d]: Circle counting took %f ms (%lu cycles)\n", 
    //         //    blockIdx.x, blockIdx.y, milliseconds, (unsigned long)(end - start));
    // }



    // // uint insertionIndex = prefixSumOutput[linearThreadIndex];
    // // for (uint i = 0; i < ci; i++) {
    // //     circleIndexBlock[insertionIndex + i] = circleIndexThread[i];
    // // }
    // // __syncthreads();

    // uint pixelX = blockMinX + threadIdx.x;
    // uint pixelY = blockMinY + threadIdx.y;
    // float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + pixelX)]);
    // float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
    //                                      invHeight * (static_cast<float>(pixelY) + 0.5f));

    // uint totalCirclesBlock = prefixSumInput[SCAN_BLOCK_DIM-1] + prefixSumOutput[SCAN_BLOCK_DIM-1];
    // float4 accum = *imgPtr;
    // if (cuConstRendererParams.sceneName == SNOWFLAKES || cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {
    //     for (uint i = 0; i < totalCirclesBlock; i++) {
    //         uint index = circleIndexBlock[i];
    //         float3 p = *(float3*)(&cuConstRendererParams.position[3*index]);
    //         shadePixelSnowflake(index, pixelCenterNorm, p, &accum);
    //     }
    // } else {
    //     for (uint i = 0; i < totalCirclesBlock; i++) {
    //         uint index = circleIndexBlock[i];
    //         float3 p = *(float3*)(&cuConstRendererParams.position[3*index]);
    //         shadePixel(index, pixelCenterNorm, p, &accum);
    //     }
    // }
    // *imgPtr = accum;

    // // Get end time in cycles
    // clock_t end_render = clock64();

    // // Only one thread per block should print
    // if (linearThreadIndex == 0) {
    //     // Convert to milliseconds (this is approximate and depends on GPU clock speed)
    //     // Assuming a clock rate of 1.5 GHz
    //     float milliseconds = (end_render - end) / 1500.0f;
    //     // // // printf("Block [%d,%d]: Rendering circles took %f ms (%lu cycles)\n", 
    //         //    blockIdx.x, blockIdx.y, milliseconds, (unsigned long)(end_render - end));
    // }




//  if (threadIdx.x == 0 || threadIdx.y == 0 || threadIdx.x == BLOCK_DIM_X -
//      1 || threadIdx.y == BLOCK_DIM_Y - 1) {
//      float4 color = make_float4(0.f, 0.f, 0.f, 0.f);
//      if (blockIdx.x == 24 && blockIdx.y == 12) {
//          color = make_float4(1.f, 1.f, 1.f, 1.f);
//      *imgPtr = color;
//  }
}

////////////////////////////////////////////////////////////////////////////////////////


CudaRenderer::CudaRenderer() {
    image = NULL;

    numCircles = 0;
    position = NULL;
    velocity = NULL;
    color = NULL;
    radius = NULL;

    cudaDevicePosition = NULL;
    cudaDeviceVelocity = NULL;
    cudaDeviceColor = NULL;
    cudaDeviceRadius = NULL;
    cudaDeviceImageData = NULL;
}

CudaRenderer::~CudaRenderer() {

    if (image) {
        delete image;
    }

    if (position) {
        delete [] position;
        delete [] velocity;
        delete [] color;
        delete [] radius;
    }

    if (cudaDevicePosition) {
        cudaFree(cudaDevicePosition);
        cudaFree(cudaDeviceVelocity);
        cudaFree(cudaDeviceColor);
        cudaFree(cudaDeviceRadius);
        cudaFree(cudaDeviceImageData);
    }
}

const Image*
CudaRenderer::getImage() {

    // need to copy contents of the rendered image from device memory
    // before we expose the Image object to the caller

    // // // printf("Copying image data from device\n");

    cudaMemcpy(image->data,
               cudaDeviceImageData,
               sizeof(float) * 4 * image->width * image->height,
               cudaMemcpyDeviceToHost);

    return image;
}

void
CudaRenderer::loadScene(SceneName scene) {
    sceneName = scene;
    loadCircleScene(sceneName, numCircles, position, velocity, color, radius);
}

void
CudaRenderer::setup() {

    int deviceCount = 0;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    // // // printf("---------------------------------------------------------\n");
    // // // printf("Initializing CUDA for CudaRenderer\n");
    // // // printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;

        // // // printf("Device %d: %s\n", i, deviceProps.name);
        // // // printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        // // // printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        // // // printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    // // // printf("---------------------------------------------------------\n");
    
    // By this time the scene should be loaded.  Now copy all the key
    // data structures into device memory so they are accessible to
    // CUDA kernels
    //
    // See the CUDA Programmer's Guide for descriptions of
    // cudaMalloc and cudaMemcpy

    cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceRadius, sizeof(float) * numCircles);
    cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);

    cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numCircles, cudaMemcpyHostToDevice);

    // Initialize parameters in constant memory.  We didn't talk about
    // constant memory in class, but the use of read-only constant
    // memory here is an optimization over just sticking these values
    // in device global memory.  NVIDIA GPUs have a few special tricks
    // for optimizing access to constant memory.  Using global memory
    // here would have worked just as well.  See the Programmer's
    // Guide for more information about constant memory.

    GlobalConstants params;
    params.sceneName = sceneName;
    params.numCircles = numCircles;
    params.imageWidth = image->width;
    params.imageHeight = image->height;
    params.position = cudaDevicePosition;
    params.velocity = cudaDeviceVelocity;
    params.color = cudaDeviceColor;
    params.radius = cudaDeviceRadius;
    params.imageData = cudaDeviceImageData;

    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

    // also need to copy over the noise lookup tables, so we can
    // implement noise on the GPU
    int* permX;
    int* permY;
    float* value1D;
    getNoiseTables(&permX, &permY, &value1D);
    cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);

    // last, copy over the color table that's used by the shading
    // function for circles in the snowflake demo

    float lookupTable[COLOR_MAP_SIZE][3] = {
        {1.f, 1.f, 1.f},
        {1.f, 1.f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, 0.8f, 1.f},
    };

    cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE);

}

// allocOutputImage --
//
// Allocate buffer the renderer will render into.  Check status of
// image first to avoid memory leak.
void
CudaRenderer::allocOutputImage(int width, int height) {

    if (image)
        delete image;
    image = new Image(width, height);
}

// clearImage --
//
// Clear's the renderer's target image.  The state of the image after
// the clear depends on the scene being rendered.
void
CudaRenderer::clearImage() {

    // 256 threads per block is a healthy number
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y);

    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        kernelClearImageSnowflake<<<gridDim, blockDim>>>();
    } else {
        kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
    }
    cudaDeviceSynchronize();
}

// advanceAnimation --
//
// Advance the simulation one time step.  Updates all circle positions
// and velocities
void
CudaRenderer::advanceAnimation() {
     // 256 threads per block is a healthy number
    dim3 blockDim(256, 1);
    dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);

    // only the snowflake scene has animation
    if (sceneName == SNOWFLAKES) {
        kernelAdvanceSnowflake<<<gridDim, blockDim>>>();
    } else if (sceneName == BOUNCING_BALLS) {
        kernelAdvanceBouncingBalls<<<gridDim, blockDim>>>();
    } else if (sceneName == HYPNOSIS) {
        kernelAdvanceHypnosis<<<gridDim, blockDim>>>();
    } else if (sceneName == FIREWORKS) { 
        kernelAdvanceFireWorks<<<gridDim, blockDim>>>(); 
    }
    cudaDeviceSynchronize();
}



typedef thrust::tuple<float,float> Float2;
typedef thrust::tuple<float,float,float,float> Float4;



struct CircleMapping {
    uint circleIndex;
    int value;
};




struct get_value
{
    __host__ __device__ get_value() {}

    __device__ int operator()(CircleMapping cm) {
        return cm.value;
    }
};








struct map_n_patches_touching_new
{
    uint patch_side_length;
    int imageWidth;

    __host__ __device__ map_n_patches_touching_new(
        uint patch_side_length,
        int imageWidth) :
        patch_side_length(patch_side_length), imageWidth(imageWidth) {}

    __device__ int operator()(int circleIndex, float radius) {
        
        float diameter = imageWidth * radius * 2;

        // Round up to nearest integer
        uint diameter_ceil = static_cast<uint>(ceil(diameter));
        uint bb_area = diameter_ceil * diameter_ceil;

        // Number of blocks needed to fully cover the circle
        // Diameter 3, block side 4 = 1 block
        uint n_patches_per_side = (diameter_ceil + patch_side_length - 1) / patch_side_length;
        // Add one more in case it sticks out
        n_patches_per_side++;

        // Square to get total number of blocks
        uint n_patches = n_patches_per_side * n_patches_per_side;

        return n_patches;
    }
};



























// How many blocks along are we? (0-indexed)
__device__ int patch_idx_for_xy(
    float x, float y, uint patches_per_row
) {
    // x,y already multiplied by image width/height
    uint x_patches_right = (x / BLOCK_DIM_X); // 1.1 = 1 along
    uint y_patches_up = (y / BLOCK_DIM_Y); 
    uint nth_patch = y_patches_up * patches_per_row + x_patches_right;

    return nth_patch;
}


// Get the nth patch index for a given circle.
// Returns -1 if the circle doesn't touch the patch.

// Patches are 0-indexed, run from 0 at bottom left to (n-1) at top right
// (but count from left to right in each row)
// Coordinates run from (x=0,y=0) at bottom left to (x=W-1,y=H-1) at top right
__device__ int get_nth_patch_idx_for_circle(
    Float2 position,
    float radius,
    uint patchNo, // Which patch to retrieve
    int imageWidth,
    int imageHeight
) {
    // float x = position.x;
    // float y = position.y;
    float x = thrust::get<0>(position);
    float y = thrust::get<1>(position);

    // Circle bbox
    float xLeft = x - radius;
    float xRight = x + radius;
    float yBtm = y - radius;
    float yTop = y + radius;

    if (xLeft < 0) xLeft = 0;
    if (xRight > 1) xRight = 1;
    if (yBtm < 0) yBtm = 0;
    if (yTop > 1) yTop = 1;


    xLeft *= imageWidth;
    xRight *= imageWidth;
    yBtm *= imageHeight;
    yTop *= imageHeight;

    // Patch indexes for all four corners
    uint imgPatchesPerRow = (imageWidth + BLOCK_DIM_X - 1) / BLOCK_DIM_X;
    uint imgPatchesPerCol = (imageHeight + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y;
    uint pi_top_left = patch_idx_for_xy(xLeft, yTop, imgPatchesPerRow);
    uint pi_top_right = patch_idx_for_xy(xRight, yTop, imgPatchesPerRow);
    uint pi_btm_left = patch_idx_for_xy(xLeft, yBtm, imgPatchesPerRow);
    uint pi_btm_right = patch_idx_for_xy(xRight, yBtm, imgPatchesPerRow);

    // Confirm patch indices covered by this circle
    uint nCircleColPatchSpan = pi_top_right - pi_top_left + 1;
    uint nCircleRowPatchSpan = (pi_top_left - pi_btm_left) / imgPatchesPerRow + 1;
    uint max_patch = imgPatchesPerRow * imgPatchesPerCol;

    // Figure out patch (px,py) within the patch grid the circle spans
    uint patchIdxCol = patchNo % nCircleColPatchSpan; // 0-indexed
    uint patchIdxRow = patchNo / nCircleColPatchSpan; 

    // Convert to global patch index
    uint patchIdx = (pi_btm_left + patchIdxCol) + \
                    (patchIdxRow * imgPatchesPerRow);

    // If greater than max patch, return -1
    if (patchIdx >= max_patch) {
        return -1;
    } else {
        return patchIdx;
    }
}


// Populate KV tuples: Value should now hold patch index this circle goes into
struct populate_kvs_new {
    int *dp_mappedCircleIds;
    int *dp_mappedPatchIds;
    int imageWidth;
    int imageHeight;

    __host__ __device__ populate_kvs_new(
        thrust::device_vector<int> &d_mappedCircleIds,
        thrust::device_vector<int> &d_mappedPatchIds,
        int imageWidth,
        int imageHeight) :
        dp_mappedCircleIds(thrust::raw_pointer_cast(d_mappedCircleIds.data())),
        dp_mappedPatchIds(thrust::raw_pointer_cast(d_mappedPatchIds.data())),
        imageWidth(imageWidth),
        imageHeight(imageHeight) {}
    
    __device__ void operator()(
        const thrust::tuple<int, int, int, Float2, float> &t
    ) {
        // Unpack tuple
        int circleIndex = thrust::get<0>(t);
        int mappingIndexSummed = thrust::get<1>(t);
        int numPatches = thrust::get<2>(t);
        //float3 position = thrust::get<3>(t);
        Float2 position = thrust::get<3>(t);
        float radius = thrust::get<4>(t);

        int idxStart = mappingIndexSummed;
        int idxEnd = idxStart + numPatches;

        // Fill KV tuples
        for (int i = idxStart; i < idxEnd; i++) {
            int patchNo = i - idxStart; // Start at 0
            dp_mappedCircleIds[i] = circleIndex;
            dp_mappedPatchIds[i] = get_nth_patch_idx_for_circle(
                position, radius, patchNo, imageWidth, imageHeight);
        }
    }
};


struct tuple_key_equals_value
{
    int value;

    __host__ __device__ tuple_key_equals_value(int value) : value(value) {}

    __host__ __device__
    bool operator()(const thrust::tuple<int, int>& t) const
    {
        return thrust::get<0>(t) == value;
    }
};


struct is_non_zero {
    __device__ int operator()(int x) const {
        return int(x != 0);
    }
};




struct make_position_tuple {
    __host__ __device__ Float2 operator()(float3 f3) const {
        return thrust::make_tuple(f3.x, f3.y);
    }
};


struct make_circle_bbox {
    __host__ __device__ Float4 operator()(
        const thrust::tuple<Float2, float> &t
    ) {
        Float2 position = thrust::get<0>(t);
        float radius = thrust::get<1>(t);

        float x = thrust::get<0>(position);
        float y = thrust::get<1>(position);
        return thrust::make_tuple(x - radius, x + radius, y - radius, y + radius); // xLeft, xRight, yBtm, yTop
    }
};


struct circle_out_of_bounds {
    int imageWidth;
    int imageHeight;

    __host__ __device__ circle_out_of_bounds(int imageWidth, int imageHeight) :
        imageWidth(imageWidth), imageHeight(imageHeight) {}

    __device__ bool operator()(
        const thrust::tuple<int, Float2, float> &t
    ) {
        int circleIndex = thrust::get<0>(t);
        Float2 position = thrust::get<1>(t);
        float radius = thrust::get<2>(t);

        float x = thrust::get<0>(position);
        float y = thrust::get<1>(position);

        return circleInBoxConservative(
            x, y, radius, 0, 1, 0, 1);
    }
};



// void // prove_dv(thrust::device_vector<int> &dv) {
//     thrust::host_vector<int> hv(dv);
//     for (int i = 0; i < dv.size(); i++) {
//         // printf("%d ", hv[i]);
//     }
//     // printf("\n");
// }







void
CudaRenderer::render() {

    // Thrust section

    // Setup: Make device versions of position, velocity, color, radius
    // thrust::device_vector<float3> d_position((float3 *)position, (float3 *)position + numCircles);
    // thrust::device_vector<float> d_radius(radius, radius + numCircles);
    // // thrust::device_vector<float3> d_velocity((float3 *)velocity, (float3 *)velocity + numCircles);
    // thrust::device_vector<float3> d_color((float3 *)color, (float3 *)color + numCircles);
    
    // // Setup shared pool allocator
    // using mr = thrust::system::cuda::universal_host_pinned_memory_resource;
    // using pool_allocator_float = thrust::mr::stateless_resource_allocator<float, mr>;
    // using pool_allocator_int = thrust::mr::stateless_resource_allocator<int, mr>;
    // pool_allocator_float allocator_float;
    // pool_allocator_int allocator_int;
    // auto exec_policy_float = thrust::cuda::par(allocator_float);
    // auto exec_policy_int = thrust::cuda::par(allocator_int);


    // Setup: Cast CUDA device pointers into Thrust vectors
    thrust::device_ptr<float3> d_position_ptr = thrust::device_pointer_cast((float3 *)cudaDevicePosition);
    
    thrust::device_ptr<float> d_radius_ptr = thrust::device_pointer_cast(cudaDeviceRadius);
    // thrust::device_vector<float, pool_allocator_float> d_radius(numCircles, allocator_float);
    thrust::device_vector<float> d_radius(numCircles);
    thrust::copy(d_radius_ptr, d_radius_ptr + numCircles, d_radius.begin());


    // Init indices for circles
    // thrust::device_vector<int, pool_allocator_int> d_circleIndices(numCircles, allocator_int);
    thrust::device_vector<int> d_circleIndices(numCircles);
    thrust::sequence(d_circleIndices.begin(), d_circleIndices.end(), 0);

    // printf("Init indices\n");
    // prove_dv(d_circleIndices);

    // Translate position float3 into more performant+useful Thrust tuple
    thrust::device_vector<Float2> d_position_tuple(numCircles);
    thrust::transform(
        d_position_ptr,
        d_position_ptr + numCircles,
        d_position_tuple.begin(),
        make_position_tuple()
    );

    // Filter out circle indices that are out of bounds
    // using circleInBoxConservative
    auto d_circleData_iter = thrust::make_zip_iterator(
        thrust::make_tuple(
            d_circleIndices.begin(),
            d_position_tuple.begin(),
            d_radius.begin()
        )
    );
    auto d_circleData_filt_last = thrust::remove_if(
        d_circleData_iter,
        thrust::make_zip_iterator(
            thrust::make_tuple(
                d_circleIndices.end(),
                d_position_tuple.end(),
                d_radius.end()
            )
        ),
        circle_out_of_bounds(image->width, image->height)
    );
    int numCirclesFilt = d_circleData_filt_last - d_circleData_iter;
    d_circleIndices.resize(numCirclesFilt);
    d_position_tuple.resize(numCirclesFilt);
    d_radius.resize(numCirclesFilt);
    // int numCirclesFilt = numCircles;


    // Check
    // // printf("Filtered out of bounds circles: From %d to %d\n", numCircles, numCirclesFilt);


    // Return early if no circles are left
    if (numCirclesFilt == 0) {
        return;
    }



    // // Unpack position and radius into bounding boxes
    // // xLeft, xRight, yBtm, yTop
    // // TODO: Convert later code to use this instead of float3
    // // Also, confirm we're working on cuda, since we're using raw position, radius
    // thrust::device_vector<Float4> d_circle_bbox(numCirclesFilt);
    // thrust::transform(
    //     thrust::make_zip_iterator(
    //         thrust::make_tuple(
    //             d_position_tuple.begin(),
    //             d_radius.begin()
    //         )
    //     ),
    //     thrust::make_zip_iterator(
    //         thrust::make_tuple(
    //             d_position_tuple.end(),
    //             d_radius.end()
    //         )
    //     ),
    //     d_circle_bbox.begin(),
    //     make_circle_bbox()
    // );

    // // Filter circles that are out of bounds




    // Populate CMs with number of patches required
    // thrust::device_vector<int, pool_allocator_int> d_n_patches_per_circle(numCirclesFilt, allocator_int); // Zeroes
    thrust::device_vector<int> d_n_patches_per_circle(numCirclesFilt);
    thrust::transform(
        d_circleIndices.begin(),
        d_circleIndices.end(),
        d_radius.begin(),
        d_n_patches_per_circle.begin(),
        map_n_patches_touching_new(BLOCK_DIM_X, image->width)
    );



    // printf("N patches required\n");
    // prove_dv(d_n_patches_per_circle);


    // Use this to get starting indices for mappings
    // thrust::device_vector<int, pool_allocator_int> d_mappingIndicesSummed(numCirclesFilt, allocator_int);
    thrust::device_vector<int> d_mappingIndicesSummed(numCirclesFilt);
    thrust::exclusive_scan(
        d_n_patches_per_circle.begin(),
        d_n_patches_per_circle.end(),
        d_mappingIndicesSummed.begin(),
        0
    ); // e.g. [2,3,1] -> [0,2,5]

    // printf("Mapping indices\n");
    // prove_dv(d_mappingIndicesSummed);



    // Allocate total space needed: Exclusive sum + last value
    int n_kvs = d_mappingIndicesSummed.back() + d_n_patches_per_circle.back();

    // First count is liberally padded: We'll remove surplus later
    // printf("Total KVs: %d\n", n_kvs);

    // thrust::device_vector<int, pool_allocator_int> d_mappedCircleIds(n_kvs, -1, allocator_int);
    // thrust::device_vector<int, pool_allocator_int> d_mappedPatchIds(n_kvs, -1, allocator_int);
    thrust::device_vector<int> d_mappedCircleIds(n_kvs);
    thrust::device_vector<int> d_mappedPatchIds(n_kvs);

    // Populate 
    auto mapping_first = thrust::make_zip_iterator(
        thrust::make_tuple( 
            d_circleIndices.begin(),
            d_mappingIndicesSummed.begin(),
            d_n_patches_per_circle.begin(),
            d_position_tuple.begin(),
            d_radius.begin()
        )
    );
    auto mapping_last = thrust::make_zip_iterator(
        thrust::make_tuple(
            d_circleIndices.end(),
            d_mappingIndicesSummed.end(),
            d_n_patches_per_circle.end(),
            d_position_tuple.end(),
            d_radius.end()
        )
    );

    // Populate KV tuples
    thrust::for_each(
        mapping_first,
        mapping_last,
        populate_kvs_new(
            d_mappedCircleIds,
            d_mappedPatchIds,
            image->width,
            image->height
        )
    );

    // Check mapped patch IDs
    // thrust::host_vector<int> h_mappedPatchIdsA(d_mappedPatchIds);
    // thrust::host_vector<int> h_mappedCircleIdsA(d_mappedCircleIds);
    // for (int i = 0; i < n_kvs; i++) {
    //     // printf("Mapping %d: Patch %d = Circle %d\n", i, h_mappedPatchIdsA[i], h_mappedCircleIdsA[i]);
    // }

    // Filter -1 KV tuples
    // Returns iterator 
    thrust::device_vector<int> minus_one(1, -1);
    auto d_nonEmptyKVs = thrust::remove_if(
        thrust::make_zip_iterator(
            thrust::make_tuple(
                d_mappedPatchIds.begin(),
                d_mappedCircleIds.begin()
            )
        ),
        thrust::make_zip_iterator(
            thrust::make_tuple(
                d_mappedPatchIds.end(),
                d_mappedCircleIds.end()
            )
        ),
        tuple_key_equals_value(-1)
    );

    // Resize
    auto iter_tuple_non_empty = d_nonEmptyKVs.get_iterator_tuple();
    auto d_mappedPatchIdsFilt = thrust::get<0>(iter_tuple_non_empty);

    int n_kvs_filt = d_mappedPatchIdsFilt - d_mappedPatchIds.begin();
    d_mappedPatchIds.resize(n_kvs_filt);
    d_mappedCircleIds.resize(n_kvs_filt);  

    // Sort ascending by patch index followed by by circle index
    auto tuple_first = thrust::make_zip_iterator(
        thrust::make_tuple(d_mappedPatchIds.begin(), d_mappedCircleIds.begin())
    );  
    auto tuple_last = thrust::make_zip_iterator(
        thrust::make_tuple(d_mappedPatchIds.end(), d_mappedCircleIds.end()));

    // Sort ascending by patch index followed by by circle index
    // This is the final mapping of patches to circles
    thrust::sort(
        tuple_first, tuple_last, 
        thrust::less<thrust::tuple<int, int>>()
    );

    // Check
    // thrust::host_vector<int> h_mappedPatchIdsFilt(d_mappedPatchIds);
    // thrust::host_vector<int> h_mappedCircleIdsFilt(d_mappedCircleIds);
    // for (int i = 0; i < n_kvs_filt; i++) {
    //     // // printf("Mapping %d: Patch %d = Circle %d\n", i, h_mappedPatchIdsFilt[i], h_mappedCircleIdsFilt[i]);
    // }

    // Next goal: Produce indices mapping patch IDs to mapping index start and count

    // Make intermediates from d_mappedPatchIds and d_mappedCircleIds

    // Calculate delta between consecutive patch IDs
    thrust::device_vector<int> d_patchDeltas(n_kvs_filt);
    thrust::adjacent_difference(
        d_mappedPatchIds.begin(),
        d_mappedPatchIds.end(),
        d_patchDeltas.begin()
    );
    d_patchDeltas[0] = 1; // Set first delta to 1 as adj difference will be large


    // Items with non-zero deltas are the start of a new patch
    // Convert this to 0s and 1s
    thrust::transform(
        d_patchDeltas.begin(), d_patchDeltas.end(),
        d_patchDeltas.begin(),
        is_non_zero()
    );

    // Ascending list of all mapping indices
    thrust::device_vector<int> d_indicesToPatches(n_kvs_filt);
    thrust::sequence(d_indicesToPatches.begin(), d_indicesToPatches.end(), 0);

    // Remove indices that are not the start of a patch
    auto d_newPatchStarts = thrust::remove_if(
        thrust::make_zip_iterator(
            thrust::make_tuple(d_patchDeltas.begin(), d_indicesToPatches.begin())
        ),
        thrust::make_zip_iterator(
            thrust::make_tuple(d_patchDeltas.end(), d_indicesToPatches.end())
        ),
        tuple_key_equals_value(0)
    );

    // Get the new patch starts
    auto d_patchStarts = thrust::get<1>(d_newPatchStarts.get_iterator_tuple());
    int n_patchStarts = d_patchStarts - d_indicesToPatches.begin();
    d_indicesToPatches.resize(n_patchStarts); // This is the final index to hand out to each kernel

    // Its size is the number of separate patches which contain work
    int n_patchesWithWork = n_patchStarts;
    
    // Check

    // Print image height and width
    // // printf("Image height: %d, Image width: %d\n", image->height, image->width);

    // thrust::host_vector<int> h_indicesToPatches(d_indicesToPatches);
    // thrust::host_vector<int> h_mappedPatchIds(d_mappedPatchIds);
    // thrust::host_vector<int> h_mappedCircleIds(d_mappedCircleIds);
    // for (int i = 0; i < h_indicesToPatches.size(); i++) {
    //     // printf("Index %d: Patch %d = Circle %d\n", h_indicesToPatches[i], h_mappedPatchIds[h_indicesToPatches[i]], h_mappedCircleIds[h_indicesToPatches[i]]);
    // }
    // printf("Total no. patches with work: %d\n", n_patchesWithWork);
    // printf("\n");


    // Setup phase complete, setup to hand over to kernels

    // Key vars: d_mappedPatchIds, d_mappedCircleIds, d_indicesToPatches, n_patchesWithWork

    // Ensure these are simple *ints in device memory
    thrust::device_vector<int> d_mappedPatchIdsSimple(d_mappedPatchIds);
    thrust::device_vector<int> d_mappedCircleIdsSimple(d_mappedCircleIds);
    thrust::device_vector<int> d_indicesToPatchesSimple(d_indicesToPatches);
    int *dp_mappedPatchIds = thrust::raw_pointer_cast(d_mappedPatchIdsSimple.data());
    int *dp_mappedCircleIds = thrust::raw_pointer_cast(d_mappedCircleIdsSimple.data());
    int *dp_indicesToPatches = thrust::raw_pointer_cast(d_indicesToPatchesSimple.data());
    int n_mappingsTotal = d_mappedPatchIds.size();

    // Number of kernels is the number of patches with work
    int n_kernels = n_patchesWithWork;


    // Launch kernels
    dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
    dim3 gridDim(n_kernels, 1, 1);
    kernelRenderCircles<<<gridDim, blockDim>>>(
        dp_mappedPatchIds,
        dp_mappedCircleIds,
        dp_indicesToPatches,
        n_patchesWithWork,
        n_mappingsTotal
    );
    cudaDeviceSynchronize();

}

