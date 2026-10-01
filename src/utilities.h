#pragma once

#include "glm/glm.hpp"

#include <algorithm>
#include <istream>
#include <iterator>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#define PI                3.1415926535897932384626422832795028841971f
#define TWO_PI            6.2831853071795864769252867665590057683943f
#define SQRT_OF_ONE_THIRD 0.5773502691896257645091487805019574556476f
#define EPSILON           0.00001f

// How paths gather light (GuiDataContainer::Integrator)
enum IntegratorMode
{
    INTEGRATOR_NAIVE = 0, // BSDF sampling only: a path picks up light only when it hits an emitter
    INTEGRATOR_NEE = 1,   // next event estimation: sample a light at every non-specular bounce
    INTEGRATOR_MIS = 2    // NEE and BSDF sampling combined with the power heuristic
};

class GuiDataContainer
{
public:
    GuiDataContainer() : TracedDepth(0) {}
    int TracedDepth;

    // Render toggles. Changing any of them restarts accumulation.
    bool SortByMaterial = false;
    bool StreamCompaction = true;
    bool Antialiasing = true;
    int Integrator = INTEGRATOR_MIS;
    bool RussianRoulette = true;
    bool MotionBlur = true;

    // Stats from the path tracer
    float AvgIterationMs = 0.0f;       // CUDA-event time per iteration, averaged since the last restart
    std::vector<int> LivePathsPerDepth; // live paths entering each bounce of the latest iteration
};

namespace utilityCore
{
    extern float clamp(float f, float min, float max);
    extern bool replaceString(std::string& str, const std::string& from, const std::string& to);
    extern glm::vec3 clampRGB(glm::vec3 color);
    extern bool epsilonCheck(float a, float b);
    extern std::vector<std::string> tokenizeString(std::string str);
    extern glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);
    extern std::string convertIntToString(int number);
    extern std::istream& safeGetline(std::istream& is, std::string& t); //Thanks to http://stackoverflow.com/a/6089413
}
