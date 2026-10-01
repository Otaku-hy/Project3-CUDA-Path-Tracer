#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/partition.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sort.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static int* dev_materialKeys = NULL; // sort keys for material sorting
static int* dev_lights = NULL;       // indices of emissive geoms, for light sampling
static int numLights = 0;

// Per-iteration GPU timing
static cudaEvent_t iterStartEvent = NULL;
static cudaEvent_t iterStopEvent = NULL;
static double totalIterationMs = 0.0;
static int timedIterations = 0;

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_materialKeys, pixelcount * sizeof(int));

    std::vector<int> lights;
    for (int i = 0; i < (int)scene->geoms.size(); i++)
    {
        if (scene->materials[scene->geoms[i].materialid].emittance > 0.0f)
        {
            lights.push_back(i);
        }
    }
    numLights = lights.size();
    cudaMalloc(&dev_lights, std::max(numLights, 1) * sizeof(int));
    cudaMemcpy(dev_lights, lights.data(), numLights * sizeof(int), cudaMemcpyHostToDevice);

    cudaEventCreate(&iterStartEvent);
    cudaEventCreate(&iterStopEvent);
    totalIterationMs = 0.0;
    timedIterations = 0;

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    cudaFree(dev_materialKeys);
    cudaFree(dev_lights);

    // pathtraceFree also runs before the first pathtraceInit, when no events exist yet
    if (iterStartEvent)
    {
        cudaEventDestroy(iterStartEvent);
        iterStartEvent = NULL;
    }
    if (iterStopEvent)
    {
        cudaEventDestroy(iterStopEvent);
        iterStopEvent = NULL;
    }

    checkCUDAError("pathtraceFree");
}

// Maps two uniform numbers to a uniform point on the unit disk with little
// distortion (Shirley-Chiu concentric mapping) [PBRTv4 A.5.1].
__device__ glm::vec2 sampleConcentricDisk(float u1, float u2)
{
    float a = 2.0f * u1 - 1.0f;
    float b = 2.0f * u2 - 1.0f;
    if (a == 0.0f && b == 0.0f)
    {
        return glm::vec2(0.0f);
    }
    float r, theta;
    if (fabsf(a) > fabsf(b))
    {
        r = a;
        theta = (PI / 4.0f) * (b / a);
    }
    else
    {
        r = b;
        theta = (PI / 2.0f) - (PI / 4.0f) * (a / b);
    }
    return r * glm::vec2(cosf(theta), sinf(theta));
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(
    Camera cam, int iter, int traceDepth, bool antialias, bool motionBlur, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);
        segment.lastPdf = 0.0f;

        // Stochastic sampled antialiasing: each iteration shoots the pixel's ray
        // through a uniformly random point inside the pixel footprint, so
        // accumulating iterations integrates (box-filters) over the pixel.
        // Depth 0 keeps this seed distinct from the shading seeds.
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        float jitterX = 0.0f;
        float jitterY = 0.0f;
        if (antialias)
        {
            thrust::uniform_real_distribution<float> u01(-0.5f, 0.5f);
            jitterX = u01(rng);
            jitterY = u01(rng);
        }

        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + jitterX - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + jitterY - (float)cam.resolution.y * 0.5f)
        );

        // Thin lens depth of field: start the ray at a random point on the lens
        // and aim it at the point where the pinhole ray crosses the plane of
        // focus. Only geometry on that plane stays sharp.
        if (cam.lensRadius > 0.0f)
        {
            thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
            glm::vec2 lens = cam.lensRadius * sampleConcentricDisk(u01(rng), u01(rng));
            glm::vec3 focusPoint = cam.position
                + segment.ray.direction * (cam.focalDistance / glm::dot(segment.ray.direction, cam.view));
            segment.ray.origin = cam.position + lens.x * cam.right + lens.y * cam.up;
            segment.ray.direction = glm::normalize(focusPoint - segment.ray.origin);
        }

        // Motion blur: the whole path sees the scene at one random moment while
        // the shutter is open, so averaging iterations averages over time.
        segment.time = 0.0f;
        if (motionBlur)
        {
            thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
            segment.time = u01(rng);
        }

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        // Terminated paths are still in the buffer when stream compaction is off
        if (pathSegment.remainingBounces <= 0)
        {
            intersections[path_index].t = -1.0f;
            return;
        }

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;
        bool tmp_outside = true;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            t = geomIntersectionTest(geom, pathSegment.ray, pathSegment.time, tmp_intersect, tmp_normal, tmp_outside);
            // TODO: add more intersection tests here... triangle? metaball? CSG?

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
                outside = tmp_outside;
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
            intersections[path_index].geomId = hit_geom_index;
            intersections[path_index].outside = outside;
        }
    }
}

// Offsets the light-sampling seed from the BSDF seed of the same bounce
#define LIGHT_SEED_OFFSET 256

// Ratio of world to object surface area on a transformed sphere around the
// point with object-space normal nObj: |det M| * |M^-T n| for the linear part M.
__device__ float sphereAreaScale(const Geom& sphere, glm::vec3 nObj)
{
    return fabsf(glm::determinant(glm::mat3(sphere.transform)))
        * glm::length(glm::mat3(sphere.invTranspose) * nObj);
}

// Area-measure pdf with which samplePointOnLight picks world point p on `light` at shutter time `time`.
__device__ float lightAreaPdf(const Geom& light, glm::vec3 p, float time)
{
    if (light.type == CUBE)
    {
        glm::vec3 s = glm::abs(light.scale);
        return 1.0f / (2.0f * (s.y * s.z + s.x * s.z + s.x * s.y));
    }
    // Uniform over the object-space sphere (area pi for radius 0.5), mapped to world space
    p -= light.motion * time;
    glm::vec3 nObj = glm::normalize(multiplyMV(light.inverseTransform, glm::vec4(p, 1.0f)));
    return 1.0f / (PI * sphereAreaScale(light, nObj));
}

// Picks a point on the surface of `light` at shutter time `time`, uniformly
// by area for cubes and by object-space area for spheres. Returns the world
// point, its outward normal and the area-measure pdf.
__device__ glm::vec3 samplePointOnLight(
    const Geom& light,
    float time,
    thrust::default_random_engine& rng,
    glm::vec3& normal,
    float& pdfArea)
{
    thrust::uniform_real_distribution<float> u01(0, 1);
    glm::vec3 pObj;
    glm::vec3 nObj;
    if (light.type == CUBE)
    {
        // Choose a face with probability proportional to its world-space area
        glm::vec3 s = glm::abs(light.scale);
        glm::vec3 faceArea(s.y * s.z, s.x * s.z, s.x * s.y);
        float pick = u01(rng) * (faceArea.x + faceArea.y + faceArea.z);
        int axis = pick < faceArea.x ? 0 : (pick < faceArea.x + faceArea.y ? 1 : 2);
        float side = u01(rng) < 0.5f ? -0.5f : 0.5f;
        pObj = glm::vec3(u01(rng) - 0.5f, u01(rng) - 0.5f, u01(rng) - 0.5f);
        pObj[axis] = side;
        nObj = glm::vec3(0.0f);
        nObj[axis] = 2.0f * side;
        pdfArea = 1.0f / (2.0f * (faceArea.x + faceArea.y + faceArea.z));
    }
    else
    {
        float z = 1.0f - 2.0f * u01(rng);
        float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));
        float phi = TWO_PI * u01(rng);
        nObj = glm::vec3(r * cosf(phi), r * sinf(phi), z);
        pObj = 0.5f * nObj;
        pdfArea = 1.0f / (PI * sphereAreaScale(light, nObj));
    }
    normal = glm::normalize(multiplyMV(light.invTranspose, glm::vec4(nObj, 0.0f)));
    return multiplyMV(light.transform, glm::vec4(pObj, 1.0f)) + light.motion * time;
}

// Shadow ray test: is anything hit between the ray origin and distance maxT?
__device__ bool occluded(const Ray& ray, float time, float maxT, Geom* geoms, int geoms_size)
{
    glm::vec3 p;
    glm::vec3 n;
    bool outside;
    for (int i = 0; i < geoms_size; i++)
    {
        float t = geomIntersectionTest(geoms[i], ray, time, p, n, outside);
        if (t > 0.0f && t < maxT)
        {
            return true;
        }
    }
    return false;
}

// MIS weight of a sample drawn with pdf `pdfA`, against a second strategy with pdf `pdfB` (beta = 2)
__device__ float powerHeuristic(float pdfA, float pdfB)
{
    float a2 = pdfA * pdfA;
    float b2 = pdfB * pdfB;
    return a2 / (a2 + b2);
}

// Next event estimation: pick a random light and a point on it, and return
// the radiance it sends to x through the BSDF, divided by the light-sampling
// pdf in solid angle. With MIS the result is weighted against the chance that
// BSDF sampling finds the same direction.
__device__ glm::vec3 sampleDirectLight(
    glm::vec3 x,
    glm::vec3 normal,
    glm::vec3 wo,
    float time,
    const Material& material,
    Geom* geoms,
    int geoms_size,
    Material* materials,
    int* lights,
    int numLights,
    bool useMis,
    thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);
    int lightIndex = lights[min((int)(u01(rng) * numLights), numLights - 1)];
    const Geom& light = geoms[lightIndex];

    glm::vec3 lightNormal;
    float pdfArea;
    glm::vec3 lightPoint = samplePointOnLight(light, time, rng, lightNormal, pdfArea);

    Ray shadowRay;
    shadowRay.origin = x + normal * 0.001f;
    glm::vec3 toLight = lightPoint - shadowRay.origin;
    float dist2 = glm::dot(toLight, toLight);
    float dist = sqrtf(dist2);
    glm::vec3 wi = toLight / dist;
    shadowRay.direction = wi;

    float cosSurface = glm::dot(normal, wi);
    float cosLight = fabsf(glm::dot(lightNormal, wi));
    if (cosSurface <= 0.0f || cosLight <= 0.0f)
    {
        return glm::vec3(0.0f);
    }

    float bsdfPdf;
    glm::vec3 f = evalBsdf(material, normal, wo, wi, bsdfPdf);
    if (bsdfPdf <= 0.0f)
    {
        return glm::vec3(0.0f);
    }

    // The sampled point itself lies at dist; stop just short of it
    if (occluded(shadowRay, time, dist - 0.001f, geoms, geoms_size))
    {
        return glm::vec3(0.0f);
    }

    // Convert the area pdf to solid angle and include the light choice
    float lightPdf = pdfArea / numLights * dist2 / cosLight;
    float weight = useMis ? powerHeuristic(lightPdf, bsdfPdf) : 1.0f;
    const Material& lightMaterial = materials[light.materialid];
    return f * lightMaterial.color * (lightMaterial.emittance * cosSurface * weight / lightPdf);
}

// Russian roulette starts at this bounce, so direct and first-bounce light stay noise-free
#define RUSSIAN_ROULETTE_DEPTH 3

// Shades every live path with its material's BSDF and generates the next ray.
// Light reaches the path in two ways: next event estimation at non-specular
// bounces (unless the integrator is naive), and BSDF-sampled rays that hit an
// emitter. With MIS both are weighted by the power heuristic; with plain NEE
// the second one only counts after a specular bounce or from the camera.
// Each pixel has exactly one path per iteration, so light is added straight
// to the image without atomics. A path terminates when it hits a light,
// escapes the scene, runs out of bounces or loses at Russian roulette.
__global__ void shadeMaterial(
    int iter,
    int depth,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    Geom* geoms,
    int geoms_size,
    int* lights,
    int numLights,
    int integrator,
    bool russianRoulette,
    glm::vec3* image)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths)
    {
        return;
    }

    PathSegment segment = pathSegments[idx];
    // Terminated paths are still in the buffer when stream compaction is off
    if (segment.remainingBounces <= 0)
    {
        return;
    }

    ShadeableIntersection intersection = shadeableIntersections[idx];
    if (intersection.t > 0.0f)
    {
        Material material = materials[intersection.materialId];
        glm::vec3 intersect = segment.ray.origin + intersection.t * segment.ray.direction;

        if (material.emittance > 0.0f)
        {
            // Hit a light: pick up its radiance and stop. Camera rays and rays
            // leaving a mirror or glass (lastPdf 0) could not have been found by
            // light sampling, so they keep full weight.
            float weight = 1.0f;
            if (integrator != INTEGRATOR_NAIVE && segment.lastPdf > 0.0f)
            {
                if (integrator == INTEGRATOR_NEE)
                {
                    weight = 0.0f; // already counted by NEE at the previous bounce
                }
                else
                {
                    float cosLight = fmaxf(fabsf(glm::dot(intersection.surfaceNormal, segment.ray.direction)), 1e-6f);
                    float lightPdf = lightAreaPdf(geoms[intersection.geomId], intersect, segment.time) / numLights
                        * intersection.t * intersection.t / cosLight;
                    weight = powerHeuristic(segment.lastPdf, lightPdf);
                }
            }
            image[segment.pixelIndex] += segment.color * (material.color * (material.emittance * weight));
            segment.remainingBounces = 0;
        }
        else if (segment.remainingBounces == 1)
        {
            // Last bounce: the next ray would never be traced, so stop without
            // sampling a light either. Every integrator then counts the same path lengths.
            segment.remainingBounces = 0;
        }
        else
        {
            if (integrator != INTEGRATOR_NAIVE && numLights > 0 && !isDeltaMaterial(material))
            {
                glm::vec3 normal = intersection.surfaceNormal;
                if (glm::dot(normal, segment.ray.direction) > 0.0f)
                {
                    normal = -normal;
                }
                thrust::default_random_engine lightRng = makeSeededRandomEngine(
                    iter, segment.pixelIndex, segment.remainingBounces + LIGHT_SEED_OFFSET);
                image[segment.pixelIndex] += segment.color * sampleDirectLight(
                    intersect, normal, -segment.ray.direction, segment.time, material,
                    geoms, geoms_size, materials, lights, numLights,
                    integrator == INTEGRATOR_MIS, lightRng);
            }

            // Seed by pixel (not buffer slot) so sorting/compaction don't change the image
            thrust::default_random_engine rng =
                makeSeededRandomEngine(iter, segment.pixelIndex, segment.remainingBounces);
            scatterRay(segment, intersect, intersection.surfaceNormal, intersection.outside, material, rng);

            // Russian roulette: keep the path with probability equal to its
            // brightest throughput channel and divide by that probability.
            // Dim paths end early, and the image stays unbiased.
            if (russianRoulette && depth >= RUSSIAN_ROULETTE_DEPTH && segment.remainingBounces > 0)
            {
                thrust::uniform_real_distribution<float> u01(0, 1);
                float survive = fminf(fmaxf(segment.color.x, fmaxf(segment.color.y, segment.color.z)), 1.0f);
                if (u01(rng) < survive)
                {
                    segment.color /= survive;
                }
                else
                {
                    segment.remainingBounces = 0;
                }
            }
        }
    }
    else
    {
        // Escaped the scene
        image[segment.pixelIndex] += segment.color * BACKGROUND_COLOR;
        segment.remainingBounces = 0;
    }

    pathSegments[idx] = segment;
}

// Material id of each intersection, used as the key for material sorting.
// Misses (and terminated paths) get -1 so they are grouped at the front.
__global__ void extractMaterialKeys(int num_paths, ShadeableIntersection* intersections, int* keys)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = intersections[idx];
        keys[idx] = intersection.t > 0.0f ? intersection.materialId : -1;
    }
}

struct IsPathAlive
{
    __host__ __device__ bool operator()(const PathSegment& path) const
    {
        return path.remainingBounces > 0;
    }
};


/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.
    //   (Now done during shading: light is added to the pixel as soon as a path finds it.)

    const bool antialias = guiData ? guiData->Antialiasing : true;
    const bool sortByMaterial = guiData ? guiData->SortByMaterial : false;
    const bool streamCompaction = guiData ? guiData->StreamCompaction : true;
    const int integrator = guiData ? guiData->Integrator : INTEGRATOR_MIS;
    const bool russianRoulette = guiData ? guiData->RussianRoulette : true;
    const bool motionBlur = guiData ? guiData->MotionBlur : true;

    std::vector<int> livePathsPerDepth;

    cudaEventRecord(iterStartEvent);

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, antialias, motionBlur, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        livePathsPerDepth.push_back(num_paths);

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_intersections
        );
        checkCUDAError("trace one bounce");
        depth++;

        // Make paths hitting the same material contiguous in memory, so threads
        // in a warp run the same BSDF code and read the same material.
        if (sortByMaterial)
        {
            extractMaterialKeys<<<numblocksPathSegmentTracing, blockSize1d>>>(
                num_paths, dev_intersections, dev_materialKeys);
            thrust::sort_by_key(thrust::device,
                dev_materialKeys, dev_materialKeys + num_paths,
                thrust::make_zip_iterator(thrust::make_tuple(dev_paths, dev_intersections)));
            checkCUDAError("sort by material");
        }

        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            depth,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_lights,
            numLights,
            integrator,
            russianRoulette,
            dev_image
        );
        checkCUDAError("shade material");

        // Move terminated paths behind the live ones; their light is already
        // in the image, so the next bounce only launches threads for live paths.
        if (streamCompaction)
        {
            PathSegment* live_end = thrust::partition(thrust::device,
                dev_paths, dev_paths + num_paths, IsPathAlive());
            num_paths = live_end - dev_paths;
            checkCUDAError("stream compaction");
        }

        iterationComplete = num_paths == 0 || depth >= traceDepth;

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    cudaEventRecord(iterStopEvent);
    cudaEventSynchronize(iterStopEvent);
    float iterationMs = 0.0f;
    cudaEventElapsedTime(&iterationMs, iterStartEvent, iterStopEvent);
    // Skip the first iteration: it includes one-time warm-up (module loading, first touch of buffers)
    if (iter > 1)
    {
        totalIterationMs += iterationMs;
        timedIterations++;
    }

    if (guiData != NULL)
    {
        guiData->AvgIterationMs = timedIterations > 0 ? (float)(totalIterationMs / timedIterations) : iterationMs;
        guiData->LivePathsPerDepth = livePathsPerDepth;
    }

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
