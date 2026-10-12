"""Compact operator overlay; optional diagnostics do not affect simulation."""
import numpy as np


def saved_recording_notice(recorder, path, outcome):
    if path is None:
        return 'No active recording | B starts recording'
    score = getattr(recorder, 'metadata', {}).get('task_evaluation', {}).get('success')
    automatic = 'PASS' if score is True else 'NOT PASSED' if score is False else 'unavailable'
    return f'SAVED {path.name} | manual: {outcome.upper()} | auto: {automatic} | B starts next'


def feeding_feedback(row, config):
    from wowskin_lab.envs.feeding_evaluation import completion_checks
    checks = completion_checks(row, config)
    descriptions = dict(
        region=('yellow spoon front outside target' if 'delivery_part_bounds_spoon_m' in config
                else 'yellow spoon tip outside target'),
        beans=f"beans {row['beans_in_spoon']}/{config['required_beans']}",
        grasp='spoon not held',
        tilt=f"tilt {row['spoon_tilt_deg']:.1f} > {config['max_tilt_deg']:g} deg",
        approach=f"approach {row['spoon_approach_angle_deg']:.1f} > {config['max_approach_angle_deg']:g} deg",
        speed=f"speed {row['spoon_bowl_speed_m_s']*1000:.0f} > {config['max_bowl_speed_m_s']*1000:g} mm/s",
        head_contact=f"head contact ({row['head_contact_count']})")
    failed = [descriptions[key] for key, passed in checks.items() if not passed]
    return ' | '.join(failed) if failed else 'All conditions met; hold still'


def render_operator_panel(labels, values, *, success=False, fault=False):
    """Colored UI bitmap; never part of acquisition-camera rendering."""
    from pathlib import Path
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    path = Path('/System/Library/Fonts/SFNSMono.ttf')
    if not path.exists():
        import matplotlib
        path = Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSansMono.ttf'
    font = ImageFont.truetype(str(path), 17)
    title_font = ImageFont.truetype(str(path), 20)
    label_width = min(240, max(124, round(max(font.getlength(x) for x in labels)+20)))
    width = min(1050, label_width+round(max(font.getlength(x) for x in values))+24)
    lines = []
    for label, value in zip(labels, values):
        parts, current = [], ''
        for word in value.split():
            candidate = (current+' '+word).strip()
            if current and font.getlength(candidate) > width-label_width-20:
                parts.append(current);current = word
            else:
                current = candidate
        parts.append(current)
        lines.append((label, parts))
    height = 20+sum(27*len(parts) for _,parts in lines)
    img = Image.new('RGB', (width, height), (20, 27, 34))
    draw = ImageDraw.Draw(img)
    border = (252, 105, 96) if fault else (79, 237, 133) if success else (66, 87, 103)
    draw.rounded_rectangle((0, 0, width-1, height-1), radius=9, outline=border, width=2)
    y = 10
    for label, parts in lines:
        is_goal = label == 'Task' and success
        if is_goal:
            draw.rectangle((5, y-3, width-6, y+27*len(parts)-3), fill=(18, 56, 38))
        draw.text((12,y), label, font=font, fill=(156,174,189))
        recording_color = ((255, 109, 101) if label == 'Record' and parts[0].startswith('REC ') else
                           (255, 200, 92) if label in ('PREPARING', 'SAVING', 'VERIFYING') else
                           (255, 200, 92) if label == 'Recording notice' and 'NOT PASSED' in ' '.join(parts) else
                           (88, 255, 144) if label == 'Recording notice' and parts[0].startswith('SAVED ') else None)
        color = ((88,255,144) if is_goal else (248,123,108) if fault else
                 recording_color or ((255,200,92) if label == 'To finish' else (230,237,242)))
        for i, part in enumerate(parts):
            draw.text((label_width,y+i*27), part, font=title_font if is_goal else font, fill=color)
        y += 27*len(parts)
    return np.asarray(img)


def operator_overlay(env, row, *, title, input_kind, input_status, recorder,
                     recording_notice, diagnostics, realtime_factor, loop_hz):
    if 'delivery_distance_m' in row:
        config = env.delivery.config
        task_text = (f"SUCCESS: delivered | beans {row['beans_in_spoon']}/{config['required_beans']}"
                     if row['success'] else
                     f"Beans {row['beans_in_spoon']}/{config['required_beans']} | "
                     f"spoon {'IN' if row['delivery_inside_target'] else 'OUT'} | "
                     f"hold {row['delivery_dwell_s']:.1f}/{config['continuous_dwell_s']:g} s")
    elif hasattr(env, 'fracture'):
        task_text = (f"BROKEN | {row.get('chip_fragment_groups', 1)} fragment groups"
                     if row.get('chip_broken_bonds', 0) else 'INTACT | saddle chip on tray')
        task_text += ' | manual outcome'
    elif hasattr(env, 'peg_score'):
        task_text = ('SUCCESS: peg inserted' if env.success else
                     f"Insert {row.get('peg_depth_m', 0.)*1000:.1f}/{env.peg_score.config['required_insertion_m']*1000:g} mm | "
                     f"offset {row.get('peg_lateral_error_m', 0.)*1000:.1f} mm | "
                     f"angle {row.get('peg_orientation_error_deg', 0.):.1f} deg | hold {row.get('peg_dwell_s', 0.):.1f} s")
    elif hasattr(env, 'routing_score'):
        task_text = ('SUCCESS: two-post route' if env.success else
                     f"Route {row.get('routing_gate_count', 0)}/6 | hold {row.get('routing_dwell_s', 0.):.1f}/"
                     f"{env.routing_score.config['scoring']['continuous_dwell_s']:g} s")
    elif hasattr(env, 'cable_layout'):
        task_text = ('SUCCESS: cable in clip' if env.success else
                     f"Clip {row.get('clip_span_count', 0)}/3 | hold {row.get('clip_dwell_s', 0.):.1f}/"
                     f"{env.clip_score.config['continuous_dwell_s']:g} s")
    else:
        if env.success:
            task_text = 'SUCCESS: cube placed and released'
        elif not row.get('cube_lifted_from_source'):
            task_text = f"Pick from amber box | lift {row.get('lift_m', 0.)*1000:.0f} mm"
        elif not row.get('cube_inside_target'):
            task_text = 'Place in green box'
        else:
            task_text = f"Release and settle | hold {row.get('cube_dwell_s', 0.):.1f} s"

    if recorder and recorder.active:
        identifier = recorder.writer.path.name
        record_text = f'REC {identifier} | {recorder.writer.count} frames | Enter save / N failure'
    elif recorder:
        record_text = 'IDLE | B starts recording'
    else:
        record_text = 'Disabled for this run'

    labels = [title.split(' /')[0], 'Task', 'Record']
    values = [f'{input_kind} | {input_status}', task_text, record_text]
    labels += ['Input frame']
    values += [env.cfg.get('teleop_frame', 'view').upper()+
               (' | A:-Y D:+Y' if env.cfg.get('teleop_frame') == 'base' and env.cfg.get('teleop_base_lateral_sign') == 1. else '')+
               ' | local workspace '+
               ('OFF' if env.cfg.get('disable_workspace_limits') or env.cfg.get('disable_cartesian_limits') else 'ON (H: bounds)')]
    if env.cfg.get('arm_controller') == 'cartesian_impedance':
        labels += ['Arm control']
        values += ['Cartesian impedance (SERL++)']
    if env.cfg.get('clip_model') == 'elastic_beam':
        labels += ['Clip model']
        values += ['Elastic beam | experimental, uncalibrated']
    if hasattr(env, 'force_grip'):
        forces = row.get('grip_force_pad_normal_force_n', [0., 0.])
        labels += ['Grip feedback [N]']
        values += [(f"{'ON' if row.get('grip_force_enabled') else 'OFF'} "
                    f"target {row.get('grip_force_target_n', 0.):.3f} "
                    f"(set {row.get('grip_force_selected_target_n', 0.):.3f}) | "
                    f"L {forces[0]:.3f} R {forces[1]:.3f} | -/= adjust")]
    if getattr(env, 'keyboard_workspace_limited', False):
        labels += ['Target boundary']
        values += ['STOP: local workspace boundary | H shows bounds']
    if env.cfg.get('finger_contact_stop', False):
        labels += ['Finger contact']
        values += [('HOLD: pads touched | Z opens / releases' if row.get('finger_contact_stop_active')
                    else 'Contact stop ON (local option)')]
    if env.cfg.get('pad_backing_contact', False):
        labels += ['Pad backing']
        values += ['Contact ON | experimental, slower']
    if env.cfg.get('disable_cartesian_limits', False):
        labels += ['Control']
        values += ['Cartesian limits OFF | centered input retains target']
    if 'delivery_distance_m' in row and not row['success']:
        labels += ['To finish']
        values += [feeding_feedback(row, env.delivery.config)]
    if diagnostics:
        from wowskin_lab.envs.pick_cube import WORKSPACE
        from wowskin_lab.teleop.target_limits import workspace_bounds
        bounds = workspace_bounds(env, WORKSPACE)
        labels += ['Workspace [world m]']
        values += ['OFF' if np.isinf(bounds).all() else
                   ' | '.join(f'{axis} {lo:g}..{hi:g}' for axis, lo, hi in zip('XYZ', *bounds))]
        model = ('Forward MLP (experimental)' if env.cfg.get('reskin_model') else
                 'Dipole + fitted gains' if env.cfg.get('magnetic_fit') else 'Dipole (unfitted)')
        labels += ['Magnetic model', 'Gripper', 'Finger load [N]', 'Mesh min V/V0',
                   'Real time', 'Physics / sensor', 'Target XYZ [m]']
        values += [model, env.cfg['gripper_control_mode'],
                   f"L {row['normal_force_n']:.3f} | R {row['passive_normal_force_n']:.3f}",
                   f"L {row['min_volume_ratio']:.3f} | R {row['passive_min_volume_ratio']:.3f}",
                   f'{realtime_factor:.2f}x | {loop_hz:.1f} updates/s',
                   f"{row.get('physics_and_sensor_wall_s', 0.)*1000:.1f} ms | "
                   f"{row.get('total_contacts', 0)} contacts / {row.get('constraint_rows', 0)} constraints",
                   ' '.join(f'{value:.3f}' for value in env.target_xyz)]
        if 'head_contact_force_n' in row:
            labels += ['Head / peak [N]', 'IK / joint margin', 'Solver']
            values += [f"{row['head_contact_force_n']:.3f} / {row['head_peak_force_n']:.3f}",
                       f"{row['ik_weighted_condition']:.2f} / {row['joint_limit_margin_deg']:.1f} deg",
                       str(row['constraint_solver'])]
        if env.predictor:
            labels += ['Pollen / physics slip']
            values += [f"{row['pollen_probability']:.3f} / {row['slip']} | transfer unvalidated"]
    else:
        labels += ['Move', 'Gripper']
        if input_kind == 'SpaceMouse':
            values += ['Push / lift: XYZ | tilt / twist: rotate',
                       'Left close | right open | both hold']
        elif input_kind == 'Pico':
            values += ['Right GRIP + move / rotate | release stops',
                       'Right TRIGGER closes | release GRIP holds']
        else:
            move = ('BASE: W/S X | A/D Y | E/Q Z' if env.cfg.get('teleop_frame', env.cfg.get('keyboard_frame')) == 'base'
                    else 'VIEW: WASD + E/Q')
            values += [move+' | I/K L/J U/O rotate', 'X close | Z open | C hold']
    labels += ['Keys']
    values += [f"R reset | V view | M cloud | H {'hide' if diagnostics else 'show'} details | P exit"]
    if recording_notice:
        labels += ['Recording notice']
        values += [recording_notice]
    if row.get('mesh_quality_warning'):
        labels += ['Mesh warning']
        values += [f"V/V0={min(row['min_volume_ratio'], row['passive_min_volume_ratio']):.3f}; "
                   'signal LIVE, accuracy unvalidated']
    return labels, values
