#!/usr/bin/env python3
"""
Script to extract mass, mass center, and inertia properties from OpenSim model file.
This script parses the bsm.osim file and extracts body properties and saves them to body_properties_output.py.
"""

import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple


def parse_osim_file(osim_path: str) -> Dict[str, Dict[str, any]]:
    """
    Parse the OpenSim model file and extract body properties.
    
    Args:
        osim_path: Path to the .osim file
        
    Returns:
        Dictionary containing body properties with structure:
        {
            "body_name": {
                "mass": float,
                "mass_center": [x, y, z],
                "inertia": [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]
            }
        }
    """
    tree = ET.parse(osim_path)
    root = tree.getroot()
    
    body_properties = {}
    
    # Find all Body elements
    for body in root.findall(".//Body"):
        body_name = body.get("name")
        if body_name:
            properties = {}
            
            # Extract mass
            mass_elem = body.find("mass")
            if mass_elem is not None:
                properties["mass"] = float(mass_elem.text)
            
            # Extract mass center
            mass_center_elem = body.find("mass_center")
            if mass_center_elem is not None:
                mass_center_values = mass_center_elem.text.split()
                properties["mass_center"] = [float(val) for val in mass_center_values]
            
            # Extract inertia tensor
            inertia_elem = body.find("inertia")
            if inertia_elem is not None:
                inertia_values = inertia_elem.text.split()
                properties["inertia"] = [float(val) for val in inertia_values]
            
            # Only add if we have all required properties
            if all(key in properties for key in ["mass", "mass_center", "inertia"]):
                body_properties[body_name] = properties
    
    return body_properties


def format_body_properties_for_python(body_properties: Dict[str, Dict[str, any]]) -> str:
    """
    Format the body properties as Python dictionaries for inclusion in custom_kin_skel.py.
    
    Args:
        body_properties: Dictionary of body properties from parse_osim_file
        
    Returns:
        Formatted Python code as string
    """
    lines = []
    lines.append("# Body mass properties extracted from bsm.osim")
    lines.append("# Mass in kg, mass center in meters, inertia tensor in kg*m^2")
    lines.append("# Inertia format: [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]")
    lines.append("")
    
    # Body masses dictionary
    lines.append("body_masses = {")
    for body_name, props in body_properties.items():
        lines.append(f'    "{body_name}": {props["mass"]:.6f},')
    lines.append("}")
    lines.append("")
    
    # Body mass centers dictionary
    lines.append("body_mass_centers = {")
    for body_name, props in body_properties.items():
        mc = props["mass_center"]
        lines.append(f'    "{body_name}": [{mc[0]:.6f}, {mc[1]:.6f}, {mc[2]:.6f}],')
    lines.append("}")
    lines.append("")
    
    # Body inertia tensors dictionary
    lines.append("body_inertias = {")
    for body_name, props in body_properties.items():
        inertia = props["inertia"]
        lines.append(f'    "{body_name}": [')
        lines.append(f'        {inertia[0]:.6f}, {inertia[1]:.6f}, {inertia[2]:.6f},  # Ixx, Iyy, Izz')
        lines.append(f'        {inertia[3]:.6f}, {inertia[4]:.6f}, {inertia[5]:.6f}   # Ixy, Ixz, Iyz')
        lines.append('    ],')
    lines.append("}")
    lines.append("")
    
    # Combined dictionary for convenience
    lines.append("body_properties = {")
    for body_name, props in body_properties.items():
        lines.append(f'    "{body_name}": {{')
        lines.append(f'        "mass": {props["mass"]:.6f},')
        mc = props["mass_center"]
        lines.append(f'        "mass_center": [{mc[0]:.6f}, {mc[1]:.6f}, {mc[2]:.6f}],')
        inertia = props["inertia"]
        lines.append(f'        "inertia": [{inertia[0]:.6f}, {inertia[1]:.6f}, {inertia[2]:.6f}, {inertia[3]:.6f}, {inertia[4]:.6f}, {inertia[5]:.6f}]')
        lines.append('    },')
    lines.append("}")
    
    return "\n".join(lines)


def main():
    """Main function to extract and display body properties."""
    osim_file = "external/SKEL/models/skel_models_v1.1/bsm.osim"
    
    try:
        # Parse the osim file
        body_properties = parse_osim_file(osim_file)
        
        print(f"Extracted properties for {len(body_properties)} bodies:")
        for body_name in body_properties.keys():
            print(f"  - {body_name}")
        
        print("\n" + "="*80)
        print("FORMATTED OUTPUT FOR custom_kin_skel.py:")
        print("="*80)
        
        # Format for Python
        formatted_output = format_body_properties_for_python(body_properties)
        print(formatted_output)
        
        # Save to file
        with open("body_properties_output.py", "w") as f:
            f.write(formatted_output)
        
        print("\n" + "="*80)
        print("Output saved to: body_properties_output.py")
        print("You can copy the content from this file to custom_kin_skel.py")
        
    except FileNotFoundError:
        print(f"Error: Could not find {osim_file}")
        print("Make sure you're running this script from the directory containing the osim file.")
    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main() 