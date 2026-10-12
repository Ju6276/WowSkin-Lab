"""Episode-level language instructions; recorded verbatim, not inferred labels."""
TASK_INSTRUCTIONS = {
    'pick_cube': 'Pick up the cube from the amber box and place it inside the green box, then release it.',
    'cable': 'Route the connected cable along the marked sides of both guide posts.',
    'cable_insert': 'Insert the cable into the clip.',
    'peg_insert': 'Insert the grasped peg into the matching socket.',
    'chip_tray': 'Pick up the saddle-shaped potato chip from the tray without breaking it.',
    'feeding': 'Move the loaded spoon towards the mouth.',
}


def episode_instruction(task, instruction=None):
    value = TASK_INSTRUCTIONS.get(task, task) if instruction is None else instruction
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Instruction must be a nonempty string')
    return value
