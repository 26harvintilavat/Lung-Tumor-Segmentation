import json
import xml.etree.ElementTree as ET
from pathlib import Path

LIDC_NS = {"lidc": "http://www.nih.gov"}

def extract_patient_id(xml_path):
    """ Convert TCIA numeric XML filename to LIDC-IDRI patient ID.Example: 
        001.xml -> LIDC-IDRI-0001
        005.xml -> LIDC-IDRI-0005
    """
    case_num = int(xml_path.stem)
    return f"LIDC-IDRI-{case_num:04d}"


def parse_lidc_xml(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    patient_id = extract_patient_id(xml_path)
    series_uid_elem = root.find(".//lidc:SeriesInstanceUid", LIDC_NS)
    if series_uid_elem is None:
        raise ValueError(f"No SeriesInstanceUid found in {xml_path}")

    series_instance_uid = series_uid_elem.text.strip()

    annotation = {
        "patient_id": patient_id,
        "series_instance_uid": series_instance_uid,
        "nodules": []
    }

    reading_sessions = root.findall(".//lidc:readingSession", LIDC_NS)

    if len(reading_sessions) == 0:
        print(f"Warning: no reading sessions in {xml_path}")

    for r_idx, session in enumerate(reading_sessions, start=1):
        radiologist_id = f"R{r_idx}"

        nodules = session.findall(".//lidc:unblindedReadNodule", LIDC_NS)

        for n_idx, nodule in enumerate(nodules, start=1):
            nodule_entry = {
                "nodule_id": f"R{r_idx}_Nodule_{n_idx}",
                "radiologist_id": radiologist_id,
                "slices": []
            }

            for roi in nodule.findall("lidc:roi", LIDC_NS):
                edges = roi.findall("lidc:edgeMap", LIDC_NS)
                if len(edges) < 3:
                    continue

                sop_uid_elem = roi.find("lidc:imageSOP_UID", LIDC_NS)
                if sop_uid_elem is None:
                        continue  # safety

                sop_uid = sop_uid_elem.text.strip()
                z_elem = roi.find("lidc:imageZposition", LIDC_NS)
                if z_elem is None:
                    continue

                z_position = float(z_elem.text)


                contour = []
                for edge in edges:
                    x = int(edge.find("lidc:xCoord", LIDC_NS).text)
                    y = int(edge.find("lidc:yCoord", LIDC_NS).text)
                    contour.append([x, y])

                nodule_entry["slices"].append({
                    "sop_uid": sop_uid,
                    "z_position": z_position,
                    "contour": contour
                })
                
            if len(nodule_entry['slices']) > 0:
                annotation['nodules'].append(nodule_entry)
    
    return annotation

def save_annotation(annotation: dict, output_dir:Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir/f"{annotation['patient_id']}.json"
    with open(output_path, 'w') as f:
        json.dump(annotation, f, indent=2)

    print(f"Saved annotation: {output_path}")

def main():
    annotations_dir = Path("data/annotations")
    output_dir = Path("data/annotations")

    xml_files = list(annotations_dir.glob("*.xml"))

    for xml_file in xml_files:
        annotation = parse_lidc_xml(xml_file)
        save_annotation(annotation, output_dir)

if __name__=="__main__":
    main()