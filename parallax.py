"""
parallax.py

Module that is used to calculate the
height of a building through the parallax
method. By triangulating the pixel vertical
displacement, we are able to find missing 
angles and lengths through the laws of sines.

Author: Amun Reddy
Date: 7-27-2025
Last Updated: 7-28-2025
"""
import math

class Point:
    def __init__(self, y, new_y, typ):
        """Stores the point and its properties."""
        self.y = y
        self.new_y = new_y
        self.typ = typ
        self.pixel_shift = abs(new_y - y)

class DroneSpecs:
    def __init__(self):
        "Stores the specs of a drone."
        self.fov_vertical = 58.5
        self.resolution = 2250

class Triangle:
    def __init__(self, point, tbDistance):
        """
        Generates triangle for the point to find the distance
        the point is from the top and bottom image.
        """
        self.point = point

        # Controls help determine what route the final angle needs to be calculated with (i.e. how the 90 degree angle will be applied)
        self.bControl = (self.point.y - (drone_specs.resolution/2))/drone_specs.resolution
        self.tControl = (self.point.new_y - (drone_specs.resolution/2))/drone_specs.resolution

        self.tbDistance = tbDistance

        # Call other nearby methods to find additional necessary variables
        self.angles()
        self.distance()
    
    def angles(self):
        """Conditionally calculates the missing angles."""
        self.pAngle = self.point.pixel_shift * (drone_specs.fov_vertical / drone_specs.resolution)

        # The temporary angles are the offset from the 90-degree angle that must be applied
        self.temp_bAngle = abs(self.bControl * drone_specs.fov_vertical)
        self.temp_tAngle = abs(self.tControl * drone_specs.fov_vertical)

        # The conditionals determine how the 90-degree angle will be applied to each of the temporary angles
        if self.bControl < 0:
            self.bAngle = 90-self.temp_bAngle

        if self.tControl < 0:
            self.tAngle = 90+self.temp_tAngle

        if self.bControl > 0:
            self.bAngle = 90+self.temp_bAngle

        if self.tControl > 0:
            self.tAngle = 90-self.temp_tAngle
                    
        # Convert to radians for final variable for correct input: sin() uses radians
        self.top_angle = math.radians(self.tAngle)
        self.bottom_angle = math.radians(self.bAngle)
        self.parallax_angle = math.radians(self.pAngle)
    
    def distance(self):
        """This method finds the distance of the two legs (the top photo to the point and the bottom)"""
        self.bpDistance = (math.sin(self.top_angle) * self.tbDistance) / math.sin(self.parallax_angle)
        self.tpDistance = (math.sin(self.bottom_angle) * self.tbDistance) / math.sin(self.parallax_angle)

def height():
    """Calculates from the two already-generated triangles, a third triangle to find the distance between the two points"""
    # b_angle and bs_angle take the angles from the two triangles to find the angles of the third
    b_angle = bottomTriangle.bottom_angle - topTriangle.bottom_angle
    bs_angle = bottomTriangle.parallax_angle + topTriangle.top_angle

    # This then uses law of sines to find the height using the two new angles above, and the leg of an the old calculated triangles
    height = (math.sin(b_angle) * topTriangle.bpDistance) / math.sin(bs_angle)
    return height

# Variable and system initialization
drone_specs = DroneSpecs()
topPoint, bottomPoint = Point(170, 236, "top"), Point(1280, 1359, "bottom")
topTriangle, bottomTriangle = Triangle(topPoint, 1.5), Triangle(bottomPoint, 1.5)

# Meter to feet conversion
height_meters = height()
height_feet = height_meters * 3.28084

# Data visualization
print(f"The height of your building is {height_meters:,.2f} meters, or {height_feet:,.2f} feet.")