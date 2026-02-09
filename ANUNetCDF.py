#!/usr/bin/env python
# 2015 Lachlan Deakin

import netCDF4
import numpy as np
import struct
import hashlib
import sys
import os
from collections import OrderedDict

class ANUNetCDFType:
    Tomo, TomoFloat, RGBA8, Segmented, DistanceMap, Label = range(0, 6)

def getANUtype(imagetype, format='NETCDF4_CLASSIC'):
    if imagetype==ANUNetCDFType.Tomo:
        if format=='NETCDF4_CLASSIC':
            dtype = np.short # FIXME: NETCDF4_CLASSIC does not support np.ushort
        else:
            dtype = np.ushort
        imtype = "tomo"
        mask = 65535
    elif imagetype==ANUNetCDFType.TomoFloat:
        dtype = np.float32
        imtype = "tomo_float"
        mask = 1.0e30
    elif imagetype==ANUNetCDFType.RGBA8:
        dtype = np.uint8
        imtype = "rgba8"
        mask = np.zeros(4)
    elif imagetype==ANUNetCDFType.Segmented:
        dtype = np.uint8
        imtype = "segmented"
        mask = 255
    elif imagetype==ANUNetCDFType.DistanceMap:
        dtype = np.float32
        imtype = "distance_map"
        mask = -2.0
    elif imagetype==ANUNetCDFType.Label:
        dtype = np.int32
        imtype = "label"
        mask = 2147483647
    else:
        print("Unknown image type")
        sys.exit(1)
    return (dtype, imtype, mask)

def getANUtypeFromName(fn):
    if fn.startswith("tomo_float"):
        return ANUNetCDFType.TomoFloat
    elif fn.startswith("tomo"):
        return ANUNetCDFType.Tomo
    elif fn.startswith("rgba8"):
        return ANUNetCDFType.RGBA8
    elif fn.startswith("segmented"):
        return ANUNetCDFType.Segmented
    elif fn.startswith("distance_map"):
        return ANUNetCDFType.DistanceMap
    elif fn.startswith("label"):
        return ANUNetCDFType.Label
    else:
        print("Could not determine image type from name")
        sys.exit(1)

class ANUNetCDF():
    def __init__(self):
        pass

    def writeFile(self, fn, array, origin, voxsz, voxunits, ID, histories, imagetype=ANUNetCDFType.Tomo):
        # Check if multiple output files are needed
        numMBytesPerFile = 256
        bytesPerFile = numMBytesPerFile*1024*1024
        slicesPerFile = min(array.shape[2], 1+(bytesPerFile-1)/(array.shape[0]*array.shape[1]))
        if slicesPerFile < array.shape[2]: # Multiple files are needed
            # Create output directory
            fn = fn.replace(".nc", "_nc")
            if not os.path.exists(fn):
                os.makedirs(fn)

            # make slicesPerFile less of a prime number...
            newValue = slicesPerFile
            divisor = [2, 4, 6, 8, 12, 16, 24, 40, 60, 80, 100, 120, 200]

            for ii in range(13):
                d = divisor[ii]
                tmpValue = d*( (slicesPerFile + d/2)/d )
                t = tmpValue * array.shape[0]*array.shape[1]
                if abs(t - bytesPerFile)/float(bytesPerFile) < 0.08:
                    newValue = tmpValue
            slicesPerFile = newValue

            fns = []
            starts = []
            sizes = []
            i_slice = 0
            while i_slice < array.shape[2]:
                fns.append(fn + "/block{:08d}.nc".format(i_slice/slicesPerFile))
                starts.append(i_slice)
                sizes.append(slicesPerFile if (array.shape[2]-i_slice)>slicesPerFile else array.shape[2]-i_slice)
                i_slice += slicesPerFile

        else: # Only a single file is needed
            fns = [fn]
            starts = [0]
            sizes = [array.shape[2]]

        for f, start, size in zip(fns, starts, sizes):
            ncfile = netCDF4.Dataset(f, mode='w', format='NETCDF4_CLASSIC')
            dtype, imtype, mask = getANUtype(imagetype)
            if isinstance(array, np.ma.MaskedArray):
                array[array.mask] = mask

            # Create dimensions
            if array.ndim==2:
                array = array[:,:,np.newaxis]
            zdim = ncfile.createDimension("{0}_zdim".format(imtype), size)
            ydim = ncfile.createDimension("{0}_ydim".format(imtype), array.shape[1])
            xdim = ncfile.createDimension("{0}_xdim".format(imtype), array.shape[0])
            ncfile.createDimension('md5sums_dim',16)

            # Create variables
            image = ncfile.createVariable(imtype, dtype, ("{0}_zdim".format(imtype),
                                                          "{0}_ydim".format(imtype),
                                                          "{0}_xdim".format(imtype)))
            md5 = ncfile.createVariable('md5_checksum', np.byte, ('md5sums_dim',))
            # data_histogram = self.createVariable('data_histogram', np.double, ('data_histogram_dim',))

            # Set attributes
            ncfile.data_description = "Raw reconstructed tomogram data <ushort>"
            ncfile.dataset_id = ID
            for key, value in histories.items():
                setattr(ncfile, key, value)
            ncfile.coordinate_origin_xyz = origin
            ncfile.voxel_size_xyz = voxsz
            ncfile.voxel_unit = voxunits
            ncfile.coord_transform = ""

            # Set the data
            image[:] = np.swapaxes(array[:,:,start:(start+size)].astype(dtype), 0, 2) # x y z -> z y x
            #image[:] = array[:,:,start:(start+size)].astype(dtype)           no transpose

            # Set the md5 hash
            md5bytestr = hashlib.md5(image[:].data).digest()
            md5[:] = struct.unpack('16B', md5bytestr)

            # Set the dimensions
            ncfile.zdim_total = array.shape[2]
            ncfile.number_of_files = len(fns)
            ncfile.zdim_range = (start, start+size-1)
            ncfile.total_grid_size_xyz = array.shape

            # Close the file
            ncfile.close()

    def readFile(self, fn, imagetype=None, format='NETCDF4'):
        ncfile = netCDF4.Dataset(fn, mode='r', format=format)
        if imagetype is None:
            imagetype = getANUtypeFromName(fn)
        dtype, imtype, mask = getANUtype(imagetype, format)

        im_data = ncfile.variables.get(imtype, None)
        if im_data is None:
            print("Could not read NetCDF file properly")
            sys.exit(1)

        array = im_data[:].astype(dtype)
        array.mask = 0 # Ensure nothing is masked, as nc is masking zero values
        if (imagetype==ANUNetCDFType.RGBA8):
            # (z, y, 4x) -> (z, y, x, 4)
            array = array.reshape((im_data.shape[0],im_data.shape[1],im_data.shape[2]/4,4))
        array = np.swapaxes(array, 0, 2) # z y x -> x y z
        array = np.ma.array(array, mask=array==mask)


        ID = getattr(ncfile,'dataset_id', 'unknown')
        origin = [int (i) for i in ncfile.coordinate_origin_xyz]
        voxsz = [float(i) for i in ncfile.voxel_size_xyz]
        voxunits = ncfile.voxel_unit

        histories = OrderedDict()
        for key in ncfile.ncattrs():
            if key.startswith("history_"):
                histories[key] = ncfile.__dict__[key]

        return (array, origin, voxsz, voxunits, ID, histories)
