#pragma once

#include "sceneStructs.h"

#include <glm/glm.hpp>

#include <thrust/random.h>

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal, 
    thrust::default_random_engine& rng);

/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 *
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 *
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * `outside` says whether the ray hit the surface from outside the primitive,
 * which picks the side of a refractive interface. Also records the pdf of the
 * sampled direction in `lastPdf` (0 for mirrors and glass).
 */
__host__ __device__ void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    bool outside,
    const Material& m,
    thrust::default_random_engine& rng);

/**
 * True for materials that scatter into a single direction (mirror, glass).
 * Light sampling cannot hit that direction, so they get no direct lighting.
 */
__host__ __device__ bool isDeltaMaterial(const Material& m);

/**
 * BSDF value f(wo, wi) for a non-delta material, and the solid-angle pdf with
 * which scatterRay would pick wi. `normal` faces wo; wo and wi point away
 * from the surface.
 */
__host__ __device__ glm::vec3 evalBsdf(
    const Material& m,
    glm::vec3 normal,
    glm::vec3 wo,
    glm::vec3 wi,
    float& pdf);
