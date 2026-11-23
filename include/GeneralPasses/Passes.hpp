#pragma once

// =============================== INFO ===============================
// Passes.hpp - ALL the passes of the engine in one include :
//
//     #include "GeneralPasses/Passes.hpp"
//
// the headers are included in the order of the frame ( the g buffer , the light , the denoisers , the post process ) . every pass is in the namespace mox::Passes . when a new pass file is
// added to this folder , it has to be added here too . the empty files ( NRCPass , PathTracing ) are included already : the pass appears when the file is written
// =============================== END ===============================

// the g buffer and the light preparation
#include "mDeferredPass.hpp"
#include "ForwardPlus.hpp"
#include "AmbientOcclusion.hpp"

// the light
#include "RayTracing.hpp"
#include "PathTracing.hpp"
#include "SHaRCPass.hpp"
#include "NRCPass.hpp"

// the denoisers
#include "BF.hpp"
#include "ReBLUR.hpp"
#include "ReLAX.hpp"

// the exposure , the tone mapping , the post process
#include "EVUpdatePass.hpp"
#include "HDRAE.hpp"
#include "HDR.hpp"
#include "PostProcess.hpp"
