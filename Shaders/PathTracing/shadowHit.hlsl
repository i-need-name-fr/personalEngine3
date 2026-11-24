struct [raypayload] ShadowRayPayLoad{
    bool inShadow : read(closesthit , miss, caller) : write(closesthit , miss , caller);
    // the distance from the origin of the ray to what stopped it ( a hit ) , the caller sets it to the length of the ray before the trace ( nothing in the way )
    float depth : read(closesthit , miss, caller) : write(closesthit , miss , caller);
};

[shader("closesthit")]
void main(inout ShadowRayPayLoad load , in BuiltInTriangleIntersectionAttributes attr){
    load.inShadow = true;
    load.depth = RayTCurrent();
}
