from two_obstacle_pilot import summarize


def test_pair_metrics_separate_route_object_from_contacted_object():
    stove = dict(
        task_id=1,
        route_obstacle="pepper_mill",
        success=True,
        joint=False,
        violation=True,
        first_contact={"obstacle": "utensil_holder", "with": "robot"},
        bowl_final_on=["stove"],
    )
    plate = dict(
        task_id=8,
        route_obstacle="utensil_holder",
        success=False,
        joint=False,
        violation=False,
        first_contact=None,
        bowl_final_on=["stove"],
    )
    other = [
        dict(stove, violation=False, joint=True, first_contact=None),
        dict(plate, success=True, joint=True, bowl_final_on=["plate"]),
    ]
    result = summarize([stove, plate, *other])
    assert result["joint_by_object"] == {"utensil_holder": 1, "pepper_mill": 1}
    assert result["wrong"] == 1 and result["placement"] == 3 and result["both_joint"] == 1
    assert result["first_contacts"] == {"utensil_holder/robot": 1}
