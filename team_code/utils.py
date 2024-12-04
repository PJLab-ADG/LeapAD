
import numpy as np
import re


def fetch_scene_summary(text):
    eles = text.split('\n')
    extracted_text = ""
    for ele in eles:
        if '-' in ele and 'far-sighted' not in ele and 'current speed' not in ele:
            extracted_text += ele.split(',')[0]
            extracted_text += '.\n'
        if 'approaching the traffic intersection' in ele:
            extracted_text += ele.split(',')[0]
            extracted_text += '.\n'
        if 'You are turning' in ele:
            extracted_text += 'You are turning.\n'
    if '</ref><ref>' in extracted_text:
        extracted_text = extracted_text.replace('</ref><ref>', '.\n-')
   
    return extracted_text


def replace_tag(text, focus_view=False):
    text = text.replace("<ref>", "")
    text = text.replace("</ref>", ",")
    text = text.replace("<box>", " ")
    text = text.replace("</box>", ".")
    if focus_view:
        text = text.replace("CAM_FRONT", "FAR-SIGHTED")
    return text


def process_information(box_with_ref, box_with_ref_filter, traffic_lights, view='front view', focus=False):
    if view == 'front view':
        aim = box_with_ref if focus else box_with_ref_filter
        if len(aim) == 0:
            return "There are no important objects in the scene."

        information = f'### Important objects in the {view} \n'
        for x in aim:
            if 'stop sign' in x:
                continue
            if 'box' in x and x['box'] is not None:
                information += '-' + x['ref'].replace("In 'CAM_FRONT', ", "").replace(' is None', '') + ', ' +f'the bounding box expressed in relative coordinates is (x1, y1, x2, y2) = {tuple(x["box"])}.' + '\n'
            else:
                information += '-' + x['ref'].replace("In 'CAM_FRONT', ", "").replace(' is None', '') + '. ' + '\n'
    elif view == 'far-sighted view':
        if len(traffic_lights) == 0:
            return "No traffic lights in the far-sighted view."

        information = f'### Traffic lights in the {view} \n'
        for x in traffic_lights:
            if 'box' in x and x['box'] is not None:
                information += '-' + x['ref'].replace("In 'CAM_FRONT', ", "").replace(' is None', '') + ', ' +f'the bounding box expressed in relative coordinates is (x1, y1, x2, y2) = {tuple(x["box"])}.' + '\n'
            else:
                information += '-' + x['ref'].replace("In 'CAM_FRONT', ", "").replace(' is None', '') + '. ' + '\n'
    return information


def fetch_all_box_with_ref(text):
    if '<ref>' in text:
        pattern = r'<ref>(.*?)</ref><box>(.*?)</box>'
        matches = re.findall(pattern, text)
        output, output_filter, traffic_light = [], [], []
        for match in matches:
            box = re.findall(r'\d+', match[1].strip())
            box = [int(x)/1000 for x in box]
            output.append({'ref': match[0].strip(), 'box': box})
            if 'traffic light' in match[0].strip():
                traffic_light.append(output[-1])
            else:
                output_filter.append(output[-1])
    else:
        output, output_filter, traffic_light = [], [], []
        matches = re.findall(r'\(\d+\)\s*(.*?)(?=\(\d+\)|$)', text)
        for match in matches:
            output.append({'ref': match.strip(), 'box': None})
            if 'traffic light' in match.strip():
                traffic_light.append(output[-1])
            else:
                output_filter.append(output[-1])

    return output, output_filter, traffic_light


def compute_angle_using_three_points(wp, p, cp):
    wpp = np.array(wp) - np.array(p)
    wpcp = np.array(wp) - np.array(cp)
    cos_value = np.sum(wpp*wpcp)/(np.linalg.norm(wpp)*np.linalg.norm(wpcp))
    value = np.arccos(cos_value)
    value = value - np.pi/2 if value > np.pi/2 else value
    return value


def compute_angle_using_four_points(p1, p2, p3, p4):
    wpp = np.array(p1) - np.array(p2)
    wpcp = np.array(p3) - np.array(p4)
    cos_value = np.sum(wpp*wpcp)/(np.linalg.norm(wpp)*np.linalg.norm(wpcp))
    value = np.arccos(cos_value)
    value = np.pi - value if value > np.pi/2 else value
    return value


def compute_angle_using_two_points(p1, p2):
    wpp = np.array(p1) - np.array(p2)
    angle_radians = np.arctan2(wpp[1], wpp[0])  
    return angle_radians
