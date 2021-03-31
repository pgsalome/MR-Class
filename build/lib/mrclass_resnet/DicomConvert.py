import subprocess as sp
import os

import shutil

class DicomConverters():

    def __init__(self, dicom, ext='.nii.gz', clean=False):
        #print ('\nStarting the DICOM conversion of {0}...'.format(dicom.split('/')[-1]))
        self.dicom_folder = dicom
        indices = [i for i, x in enumerate(dicom) if x == "/"]
        self.path= dicom[0:indices[-1]]
        self.filename=dicom[indices[-1] +1 :]
        self.ext = ext
        
       
        

    def dcm2niix_converter(self, compress=True):

        outpath=self.path
        #outname = os.path.join(outpath, self.filename) + self.ext
        
        if compress:
            cmd = ("dcm2niix -s y -o {0} -f {1} -z y {2} ".format(outpath, self.filename,
                                                            self.dicom_folder))
        else:
            cmd = ("dcm2niix -s y -o {0} -f {1} {2} ".format(outpath, self.filename,
                                                       self.dicom_folder))
        sp.check_output(cmd, shell=True)
        #pdb.set_trace()
        file = [ item for item in os.listdir(outpath) if '.nii.gz' in item ]
        for f in file:
            if self.filename in f:
                outname=os.path.join(outpath, f)
                shutil.move(self.dicom_folder,outname[0:-7])
                
                break
            
        
        #print('Conversion done!\n')
        return outname
