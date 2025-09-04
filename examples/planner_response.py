def create_vlm_constraint_response(object_list, safety_constraint_list, target_constraint_list):

    fields = {}
    for i in range(len(object_list)):
        fields[f"explanation_obj_{object_list[i]}"] = (str, ...)
        fields[f"{object_list[i]}"] = (safety_constraint_list, ...)

    SafetyConstraintsPerObject = create_model("SafetyConstraintsPerObject", **fields)
    
    class ConstraintResponse(BaseModel):
        safety_constraints_per_object: SafetyConstraintsPerObject
        target_region: target_constraint_list
        target_description: str
        image_description: str

    return ConstraintResponse