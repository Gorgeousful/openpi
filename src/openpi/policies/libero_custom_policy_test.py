from openpi.policies import libero_custom_policy


def test_format_grounding_with_loc_tokens():
    grounding = '[("robotic gripper", [93, 24, 163, 56]), ("blue and white drawer", None)]'

    formatted = libero_custom_policy._format_grounding_with_loc_tokens(
        grounding,
        image_height=224,
        image_width=224,
    )

    assert formatted == "<loc0110><loc0425><loc0256><loc0744> robotic gripper; none blue and white drawer"


def test_strip_wrapping_brackets():
    assert (
        libero_custom_policy._strip_wrapping_brackets("[robotic gripper, drawer handle, drawer]")
        == "robotic gripper, drawer handle, drawer"
    )
    assert (
        libero_custom_policy._strip_wrapping_brackets("(robotic gripper, drawer handle, drawer)")
        == "robotic gripper, drawer handle, drawer"
    )
    assert (
        libero_custom_policy._strip_wrapping_brackets("robotic gripper, drawer handle, drawer")
        == "robotic gripper, drawer handle, drawer"
    )
