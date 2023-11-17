from exif import Image

# References: https://www.digicamdb.com/
# These might be slightly off...
sensorWidthMMDict = {'Canon PowerShot A10': 5.33, 'Canon PowerShot A710 IS' : 5.75, 'Canon PowerShot A510': 5.33, 'Galaxy Nexus': None, 'V5V': None }
sensorHeightMMDict = {'Canon PowerShot A10': 4, 'Canon PowerShot A710 IS' : 4.32, 'Canon PowerShot A510': 4, 'Galaxy Nexus': None, 'V5V': None }

def extractEXIFData( imgPath ):
    # Read EXIF data
    # Inelegant; requires loading the image twice
    with open(imgPath, 'rb') as image_file:
        my_image = Image(image_file)

    make = None
    cameramodel = None
    focalLengthMM = None
    focalLengthMM_35MMEquivalent = None

    try:
        cameramodel = my_image['model']
    except:
        print( imgPath + " - missing camera model name (e.g., Canon PowerShot A10).")

    try:
        focalLengthMM = my_image['focal_length']
    except:
        print( imgPath + " - missing focal length.")
    
    return make, cameramodel, focalLengthMM, focalLengthMM_35MMEquivalent


def getSensorWidthMM( make ):
    if make in sensorWidthMMDict:
        return sensorWidthMMDict[make]
    else:
        return None


def getFocalLengthInPixels( focalLengthMM, imageWidthPixels, sensorWidthMM ):
    if focalLengthMM is None or sensorWidthMM is None:
        return None

    # Draw a point representing the center of projection, and draw a virtual sensor plane, such that the center of projection and edges of the sensor form a triangle.
    # - The focal length is the perpendicular distance from the center of projection to the sensor plane
    # - The ratio between the focal length in pixels (unknown) and the focal length in mm  is the same as  the ratio between the image width in pixels and the sensor width in mm
    
    return 1 # TODO To replace
