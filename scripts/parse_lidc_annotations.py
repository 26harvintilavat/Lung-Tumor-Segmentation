import json
import xml.etree.ElementTree as ET
from pathlib import Path

LIDC_NS = {"lidc": "http://www.nih.gov"}

def parse_lidc_xml(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    patient_id = xml_path.stem

    annotation = {
        "patient_id": patient_id,
        "nodules": []
    }

    reading_sessions = root.findall(".//lidc:readingSession", LIDC_NS)

    for r_idx, session in enumerate(reading_sessions, start=1):
        radiologist_id = f"R{r_idx}"

        nodules = session.findall(".//lidc:unblindedReadNodule", LIDC_NS)

        for n_idx, nodule in enumerate(nodules, start=1):
            nodule_entry = {
                "nodule_id": f"Nodule_{n_idx}",
                "radiologist_id": radiologist_id,
                "slices": []
            }

            rois = nodule.findall(".//lidc:roi", LIDC_NS)

            for roi in rois:
                z_pos = float(roi.find("lidc:imageZposition", LIDC_NS).text)

                contour = []
                for edge in roi.findall(".//lidc:edgeMap", LIDC_NS):
                    x = int(edge.find("lidc:xCoord", LIDC_NS).text)
                    y = int(edge.find("lidc:yCoord", LIDC_NS).text)
                    contour.append([x, y])

                nodule_entry['slices'].append({
                    "z_position": z_pos,
                    "contour": contour
                })
            
            annotation["nodules"].append(nodule_entry)
    
    return annotation

def save_annotation(annotation: dict, output_dir=Path):
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