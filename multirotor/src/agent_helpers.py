def get_true_kinematic_state(**kwargs):
    return kwargs['agent']._client.getMultirotorState().kinematics_true
