from lib.device import Camera
from lib.processors import findFaceGetPulse
from lib.interface import plotXY, imshow, waitKey,destroyWindow, moveWindow
import numpy as np      
import datetime

class getPulseApp(object):
    """
    Main  application that finds a face in a Video stream, then isolates the
    forehead.    Then the  green-light intensity in the forehead region is gathered 
    over time, and based on that person's pulse is determined.
    """
    def __init__(self):
        #Imaging device or video stream in mpeg
        #videostream)
        self.camera = Camera(camera=0) #first camera by default
        self.w,self.h = 0,0
        self.pressed = 0
        
        #Image analysis such as face detection, forehead isolation, time series collection
        #heart-beat detection, etc. 

        
        self.processor = findFaceGetPulse(bpm_limits = [60,130],
                                          data_spike_limit = 2500.,
                                          face_detector_smoothness = 10.)  

        #Initialize parameters for the cardiac data plot
        self.bpm_plot = False
        self.plot_title = "Cardiac Cycle"

        #Maps keystrokes to specified methods
       
        self.key_controls = {"s" : self.toggle_search,
                             "d" : self.toggle_display_plot,
                             "f" : self.write_csv}
        
    def write_csv(self):
        """
        Writes current data to a csv file
        """
        bpm = " " + str(int(self.processor.measure_heart.bpm))
        fn = str(datetime.datetime.now()).split(".")[0] + bpm + " BPM.csv"
        
        data = np.array([self.processor.fft.times, 
                         self.processor.fft.samples]).T
        np.savetxt(fn, data, delimiter=',')
        


    def toggle_search(self):
        """
		Quality of pulse detection improves significantly by 
		freezing the location of the forehead being analyzed.
        
        """
        state = self.processor.find_faces.toggle()
        if not state:
        	self.processor.fft.reset()
        print "face detection lock =",not state

    def toggle_display_plot(self):
        """
        Toggles the data display.
        """
        if self.bpm_plot:
            print "bpm plot disabled"
            self.bpm_plot = False
            destroyWindow(self.plot_title)
        else:
            print "bpm plot enabled"
            self.bpm_plot = True
            self.make_bpm_plot()
            moveWindow(self.plot_title, self.w,0)

    def make_bpm_plot(self):
        """
        Creates output for data display
        """
        plotXY([[self.processor.fft.times, 
                 self.processor.fft.samples],
                [self.processor.fft.even_times[4:-4], 
                 self.processor.measure_heart.filtered[4:-4]],
                [self.processor.measure_heart.freqs, 
                 self.processor.measure_heart.fft]], 
               labels = [False, False, True],
               showmax = [False,False, "bpm"], 
               label_ndigits = [0,0,0],
               showmax_digits = [0,0,1],
               skip = [3,3,4],
               name = self.plot_title, 
               bg = self.processor.grab_faces.slices[0])

    def key_handler(self):    
        """
        Handling Keys being pressed by user while capturing camera input
        """

        self.pressed = waitKey(10) & 255 
        if self.pressed == 27: #exit program on 'esc'
            print "exiting..."
            self.camera.cam.release()
            exit()

        for key in self.key_controls.keys():
            if chr(self.pressed) == key:
                self.key_controls[key]()

    def main_loop(self):
        """
        Single iteration of the application's main loop.
        """
        # Get current image frame from the camera
        frame = self.camera.get_frame()
        self.h,self.w,_c = frame.shape
        

        #display unaltered frame
        #imshow("Orig",frame)

        #set current image frame to the processor's input
        self.processor.frame_in = frame
        #process the image frame to perform all needed analysis
        self.processor.run()
        #collect the output frame for display
        output_frame = self.processor.frame_out

        #show the processed/annotated output frame
        imshow("VitalSight",output_frame)

        #create and/or update the raw data display if needed
        if self.bpm_plot:
            self.make_bpm_plot()

        #handle any key presses
        self.key_handler()

if __name__ == "__main__":    
    App = getPulseApp()
    while True:
        App.main_loop()
