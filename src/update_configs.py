import json
import argparse


def modify_json(input_file):
    # Read the JSON file
    with open(input_file, 'r') as file:
        data = json.load(file)

    # Replace "aabb_scale" value with 1
    data["aabb_scale"] = 1

    # Add "scale": 0.5 and "offset": [0.5, 0.5, 0.5] after "aabb_scale"
    # For some scenes "0.20" to have watertide meshes. see c
    data["scale"] = 0.20
    data["offset"] = [0.5, 0.5, 0.5]

    # For each "file_path" in "frames", replace "images" with "rgba" in their text
    for frame in data["frames"]:
        f = frame["file_path"].split('/')[-1]
        frame["file_path"] = './rgba/' + f.replace('.jpg', '.png')

    # Write the modified data back to the JSON file
    with open(input_file, 'w') as file:
        json.dump(data, file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modify JSON file")
    parser.add_argument("input_file", help="Path to the input JSON file")
    args = parser.parse_args()

    modify_json(args.input_file)
