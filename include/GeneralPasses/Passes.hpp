#pragma once

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

#include "EVUpdatePass.hpp"
#include "HDRAE.hpp"
#include "HDR.hpp"
#include "Volumentric.hpp"
#include "PostProcess.hpp"
#include "DLSSPass.hpp"
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
#include "NKPDInferenceDiffuse.hpp"