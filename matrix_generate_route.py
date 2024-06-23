import xml.etree.ElementTree as ET
from xml.dom import minidom

def create_xml(volumes):
    '''
    ns: straight flow from north to south
    nw: right flow from north to west
    ne: left flow from north to east
    sn: straight flow from south to north
    sw: left flow from south to west
    se: right flow from south to east
    we: straight flow from west to east
    ws: right flow from west to south
    wn: left flow from west to north
    ew: straight flow from east to west
    es: left flow from east to south
    en: right flow from east to north
    '''
    ns, nw, ne, sn, sw, se, we, ws, wn, ew, es, en = volumes
    # 创建根元素
    root = ET.Element("routes")

    # 添加vType元素
    vType = ET.SubElement(root, "vType", accel="2.6", decel="4.5", id="CarA", length="5.0", minGap="2.5", maxSpeed="55.55", sigma="0.5")

    # 添加route元素
    routes = ["n_t t_s", "n_t t_w", "n_t t_e", "s_t t_n", "s_t t_w", "s_t t_e", "w_t t_e", "w_t t_s", "w_t t_n", "e_t t_w", "e_t t_s", "e_t t_n"]
    for i, route in enumerate(routes, start=1):
        ET.SubElement(root, "route", id=f"route{i:02d}", edges=route)

    # 添加flow元素并替换period值
    periods = [ns, nw, ne, sn, sw, se, we, ws, wn, ew, es, en]
    for i, period in enumerate(periods, start=1):
        ET.SubElement(root, "flow", id=f"flow{i:02d}", begin="0", end="100000", period=f"exp({period})", route=f"route{i:02d}", type="CarA", color="1,1,0")

    # 创建树结构并进行格式化
    rough_string = ET.tostring(root, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_string = reparsed.toprettyxml(indent="  ")

    # 将格式化后的XML写入文件
    with open("output.rou.xml", "w", encoding="utf-8") as f:
        f.write(pretty_string)

# 示例调用
capacity_straight = 2080
capacity_left = 1411
capacity_right = 1513
green_time_proportion = (30 - 4) / 120
convert_to_seconds = 1/3600
n_propoertion = 0.9
s_propoertion = 0.6
w_propoertion = 0.9
e_propoertion = 0.6
volumes = [capacity_straight * n_propoertion, capacity_right * n_propoertion, capacity_left * n_propoertion, 
           capacity_straight * s_propoertion, capacity_left * s_propoertion, capacity_right * s_propoertion, 
           capacity_straight * w_propoertion, capacity_right * w_propoertion, capacity_left * w_propoertion, 
           capacity_straight * e_propoertion, capacity_left * e_propoertion, capacity_right * e_propoertion]

volumes = [round(volume * green_time_proportion * convert_to_seconds, 3) for volume in volumes]
create_xml(volumes)
