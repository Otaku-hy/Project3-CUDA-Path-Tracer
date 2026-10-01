#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

// Below this roughness a reflective material is treated as a perfect mirror
#define MIN_GLOSSY_ROUGHNESS 0.01f

// GGX (Trowbridge-Reitz) microfacet distribution. alpha2 = alpha^2, with alpha = roughness^2.
__host__ __device__ static float ggxD(float cosH, float alpha2)
{
    float d = cosH * cosH * (alpha2 - 1.0f) + 1.0f;
    return alpha2 / (PI * d * d);
}

// Smith masking term of GGX for one direction
__host__ __device__ static float ggxG1(float cosV, float alpha2)
{
    return 2.0f * cosV / (cosV + sqrtf(alpha2 + (1.0f - alpha2) * cosV * cosV));
}

__host__ __device__ static glm::vec3 fresnelSchlick(glm::vec3 f0, float cosTheta)
{
    float m = glm::clamp(1.0f - cosTheta, 0.0f, 1.0f);
    float m2 = m * m;
    return f0 + (glm::vec3(1.0f) - f0) * (m2 * m2 * m);
}

// Unpolarized Fresnel reflectance of a smooth dielectric interface [PBRTv4 9.3.5].
// cosI is measured on the incident side; eta = n_transmitted / n_incident.
__host__ __device__ static float fresnelDielectric(float cosI, float eta)
{
    float sin2T = (1.0f - cosI * cosI) / (eta * eta);
    if (sin2T >= 1.0f)
    {
        return 1.0f; // total internal reflection
    }
    float cosT = sqrtf(1.0f - sin2T);
    float rParallel = (eta * cosI - cosT) / (eta * cosI + cosT);
    float rPerpendicular = (cosI - eta * cosT) / (cosI + eta * cosT);
    return 0.5f * (rParallel * rParallel + rPerpendicular * rPerpendicular);
}

__host__ __device__ bool isDeltaMaterial(const Material& m)
{
    return m.hasRefractive > 0.0f || (m.hasReflective > 0.0f && m.roughness < MIN_GLOSSY_ROUGHNESS);
}

__host__ __device__ glm::vec3 evalBsdf(
    const Material& m,
    glm::vec3 normal,
    glm::vec3 wo,
    glm::vec3 wi,
    float& pdf)
{
    pdf = 0.0f;
    float cosO = glm::dot(normal, wo);
    float cosI = glm::dot(normal, wi);
    if (isDeltaMaterial(m) || cosO <= 0.0f || cosI <= 0.0f)
    {
        return glm::vec3(0.0f);
    }

    if (m.hasReflective > 0.0f)
    {
        // GGX microfacet reflection: f = D G F / (4 cos_o cos_i)
        float alpha = m.roughness * m.roughness;
        float alpha2 = alpha * alpha;
        glm::vec3 h = glm::normalize(wo + wi);
        float cosH = glm::dot(normal, h);
        float woDotH = glm::dot(wo, h);
        float d = ggxD(cosH, alpha2);
        pdf = d * cosH / (4.0f * woDotH);
        return fresnelSchlick(m.color, woDotH) * (d * ggxG1(cosO, alpha2) * ggxG1(cosI, alpha2) / (4.0f * cosO * cosI));
    }

    pdf = cosI / PI;
    return m.color / PI;
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    bool outside,
    const Material &m,
    thrust::default_random_engine &rng)
{
    // Offset along the normal so the new ray does not re-hit the surface it
    // leaves from (walls in the sample scenes are 0.01 units thick).
    const float rayOffset = 0.001f;

    // Treat surfaces as two-sided: shade with the normal facing the incoming ray.
    glm::vec3 incident = pathSegment.ray.direction;
    if (glm::dot(normal, incident) > 0.0f)
    {
        normal = -normal;
    }
    glm::vec3 wo = -incident;
    pathSegment.remainingBounces--;

    if (m.hasRefractive > 0.0f)
    {
        // Smooth dielectric. Reflect with probability F (Fresnel) and refract
        // otherwise; picking each branch with its own weight makes the throughput
        // factor F / F = 1. Total internal reflection gives F = 1.
        thrust::uniform_real_distribution<float> u01(0, 1);
        float etaIncident = outside ? 1.0f : m.indexOfRefraction;
        float etaTransmitted = outside ? m.indexOfRefraction : 1.0f;
        float f = fresnelDielectric(glm::dot(wo, normal), etaTransmitted / etaIncident);
        if (u01(rng) < f)
        {
            pathSegment.ray.direction = glm::reflect(incident, normal);
            pathSegment.ray.origin = intersect + normal * rayOffset;
        }
        else
        {
            pathSegment.ray.direction = glm::normalize(glm::refract(incident, normal, etaIncident / etaTransmitted));
            pathSegment.ray.origin = intersect - normal * rayOffset;
            pathSegment.color *= m.color;
        }
        pathSegment.lastPdf = 0.0f;
        return;
    }

    if (m.hasReflective > 0.0f && m.roughness < MIN_GLOSSY_ROUGHNESS)
    {
        // Perfect mirror
        pathSegment.ray.direction = glm::reflect(incident, normal);
        pathSegment.ray.origin = intersect + normal * rayOffset;
        pathSegment.color *= m.color;
        pathSegment.lastPdf = 0.0f;
        return;
    }

    if (m.hasReflective > 0.0f)
    {
        // Glossy GGX reflection. Sample a microfacet normal h with pdf D(h) cos(theta_h)
        // and mirror wo about it; the pdf of the reflected direction is
        // D cos(theta_h) / (4 wo.h), so f cos(theta_i) / pdf = F G wo.h / (cos_o cos(theta_h)).
        thrust::uniform_real_distribution<float> u01(0, 1);
        float alpha = m.roughness * m.roughness;
        float alpha2 = alpha * alpha;
        float u = u01(rng);
        float cosH = sqrtf((1.0f - u) / (1.0f + (alpha2 - 1.0f) * u));
        float sinH = sqrtf(fmaxf(0.0f, 1.0f - cosH * cosH));
        float phi = TWO_PI * u01(rng);

        glm::vec3 notNormal = fabsf(normal.x) < SQRT_OF_ONE_THIRD ? glm::vec3(1, 0, 0) : glm::vec3(0, 1, 0);
        glm::vec3 tangent = glm::normalize(glm::cross(normal, notNormal));
        glm::vec3 bitangent = glm::cross(normal, tangent);
        glm::vec3 h = sinH * cosf(phi) * tangent + sinH * sinf(phi) * bitangent + cosH * normal;

        glm::vec3 wi = glm::reflect(incident, h);
        float cosO = glm::dot(normal, wo);
        float cosI = glm::dot(normal, wi);
        float woDotH = glm::dot(wo, h);
        if (cosI <= 0.0f || cosO <= 0.0f || woDotH <= 0.0f)
        {
            // Reflected into the surface: the path carries no more energy
            pathSegment.color = glm::vec3(0.0f);
            pathSegment.remainingBounces = 0;
            return;
        }

        pathSegment.ray.direction = wi;
        pathSegment.ray.origin = intersect + normal * rayOffset;
        pathSegment.color *= fresnelSchlick(m.color, woDotH) * (ggxG1(cosO, alpha2) * ggxG1(cosI, alpha2) * woDotH / (cosO * cosH));
        pathSegment.lastPdf = ggxD(cosH, alpha2) * cosH / (4.0f * woDotH);
        return;
    }

    // Ideal diffuse (Lambertian) BSDF. With f = albedo / pi and a
    // cosine-weighted pdf = cos(theta) / pi, the throughput update
    // f * cos(theta) / pdf reduces to multiplying by the albedo.
    pathSegment.ray.direction = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
    pathSegment.ray.origin = intersect + normal * rayOffset;
    pathSegment.color *= m.color;
    pathSegment.lastPdf = glm::dot(normal, pathSegment.ray.direction) / PI;
}
