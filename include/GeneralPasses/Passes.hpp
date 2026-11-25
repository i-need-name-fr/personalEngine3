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
#include "mRTPass.hpp"
#include "ForwardPlus.hpp"
#include "AmbientOcclusion.hpp"

// the light
#include "RayTracing.hpp"
#include "PathTracing.hpp"
#include "SHaRCPass.hpp"

// the denoisers
#include "BF.hpp"
#include "ReBLUR.hpp"
#include "ReLAX.hpp"

// the exposure , the tone mapping , the post process , the UI ( the last : it draws over the colour of HDR )
#include "EVUpdatePass.hpp"
#include "HDRAE.hpp"
#include "HDR.hpp"
#include "Volumentric.hpp"
#include "PostProcess.hpp"
#include "GUIPass.hpp"
#include "NRCTrain.hpp"
#include "NRCBack.hpp"
#include "NRCApply1.hpp"
#include "NRCApply2.hpp"
#include "NRCApply3.hpp"
#include "NKPDTrain.hpp"
#include "NKPDBack.hpp"
#include "NKPDApply1.hpp"
#include "NKPDApply2.hpp"
#include "NKPDApply3.hpp"
#include "RayTracingIndirect.hpp"
#include "PathTracingIndirect.hpp"
#include "NKPDInference.hpp"