 #!/usr/bin/env python

# from fcntl import F_FULLFSYNC
import tkinter as tk
import ttkbootstrap as ttk
import cv2
from PIL import Image, ImageTk
from collections import deque
from picamera2 import Picamera2
import threading
import pickle
from tkinter import filedialog
from tkinter import simpledialog
import serial
import time
from libcamera import Transform
import RPi.GPIO as GPIO
import numpy as np


class app(tk.Tk):
    def __init__(self):
        super().__init__()
        # self.window = window
        self.minsize(200, 100)
        self.title("Inspection System")

        # A frame combines movement and waypoints
        self.Mov_Way_frame = ttk.Frame(self)
        self.Mov_Way_frame.pack(side='left')

        # Inspection result label
        self.inspection_result = tk.StringVar(value='NULL')
        self.Result_Label = ttk.Label(self, textvariable=self.inspection_result, font='Calibri 8 bold',
                                      foreground='black')

        # Initialise memory to store all data
        self.memory = deque()

        self.memory_index = 0
        self.feature_point_index = 0
        self.Inspect_mode_off = False
        self.inspect_flag = False
        self.train_flag = False
        self.Inspection_Thread = threading.Thread(target = self.test)
        self.Inspection_Thread.start()
        
        # Initialise all sub classes
        self.waypoint = Waypoint(self.Mov_Way_frame)
        self.waypoint.Waypoint_table.bind('<Delete>', self.delete_waypoint_item)
        self.movement = Movement(self.Mov_Way_frame, self)
        self.camera = CameraView(self)
        self.movement_thread = threading.Thread(target = self.movement.send)
        self.movement_thread.start()

        # Three buttons
        self.save_button = ttk.Button(self, text='Save', command=self.save_point, bootstyle='success-outline')
        self.train_button = ttk.Button(self, text='Train', command=self.train_point, bootstyle='success-outline')
        self.finish_train_button = ttk.Button(self, text='Finish Training', command=self.generate_adaptive_threshold, bootstyle='success-outline')
        self.inspection_button = ttk.Button(self, text='Inspect', command=self.start_inspection, bootstyle='success-outline')
        self.pause_inspection_button = ttk.Button(self, text='Pause Inspection', command=self.pause_inspection, bootstyle='success-outline')
        self.initialisation_button = ttk.Button(self, text='Initialisation', command=self.initialise, bootstyle='success-outline')

        # Menu for saving file and opening the saved files 
        self.menu = tk.Menu(self)
        self.file_menu = tk.Menu(self.menu, tearoff=False)
        self.file_menu.add_command(label='Open', command=self.OpenFile)
        self.file_menu.add_command(label='Save', command=self.SaveFile)
        self.file_menu.add_command(label='New', command=self.NewFile)
        self.menu.add_cascade(label='File', menu=self.file_menu)
        self.configure(menu=self.menu)

        # Pack
        self.save_button.pack(side='left', padx=5, pady=5)
        self.train_button.pack(side='left', padx=5, pady=5)
        self.finish_train_button.pack(side='left', padx=5, pady=5)
        self.inspection_button.pack(side='left', padx=5, pady=5)
        self.pause_inspection_button.pack(side='left', padx=5, pady=5)
        self.initialisation_button.pack(side='left', padx=5, pady=5)
        self.Result_Label.pack(side='right')

        # Interrupt
        self.Turntable_switch = 26
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.Turntable_switch, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.add_event_detect(self.Turntable_switch, GPIO.RISING, callback=self.home_turntable, bouncetime=4000)

        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.mainloop()

    # When close the GUI
    def on_closing(self):
        GPIO.cleanup()
        self.movement.mega.close()
        self.Inspect_mode_off = True    # Terminate the inspection thread
        self.destroy()

    def delete_waypoint_item(self, _):

        i = self.waypoint.Waypoint_table.selection()[0]
        self.waypoint.Waypoint_table.delete(i)
        del self.memory[int(i)]

    def initialise(self):
        for i in range(len(self.memory)):
            self.waypoint.Waypoint_table.item(str(i), values=(self.memory[i][11], 'Unchecked', str(self.memory[i][0])+" "+str(self.memory[i][1])+" "+str(self.memory[i][2])+" "+str(self.memory[i][3])))
        self.Result_Label.configure(foreground='black')
        self.inspection_result.set('NULL')
        self.memory_index = 0

    def home_turntable(self, channel):
        print("Triggered")
        self.movement.mega.write(("M410" + "\n").encode('utf-8'))
        time.sleep(1)
        self.movement.mega.write(("G92 E0" + "\n").encode('utf-8'))
        # Rotate in reverse
        self.movement.mega.write(("G0 E-4" + "\n").encode('utf-8'))
        self.movement.mega.read_until("ok").decode('utf-8').rstrip()
        self.movement.mega.write(("M400" + "\n").encode('utf-8'))
        self.movement.mega.write(("M114" + "\n").encode('utf-8'))
        self.movement.wait_until_movement()
        # Reset
        self.movement.mega.write(("G92 E0" + "\n").encode('utf-8'))
        self.movement.reached = True
        self.movement.rotate = 0

    def save_point(self):

        # Not saving template image
        template_gray = []

        if self.camera.start_x is None: # Way point
            focus_length = None
            start_height = None
            end_height = None
            start_width = None
            end_width = None
            threshold = None
        else:                           # Feature point
            # Not taking photo, but saving other info, like roi.
            self.camera.camera.autofocus_cycle()
            focus_length = self.camera.camera.capture_metadata()["LensPosition"]
            self.camera.camera.set_controls({"AfMode": 0, "LensPosition": focus_length})
            start_height = int((self.camera.fullRes[1]/self.camera.lowerRes[1])*self.camera.start_y)
            end_height = int((self.camera.fullRes[1]/self.camera.lowerRes[1])*self.camera.end_y)
            start_width = int((self.camera.fullRes[0]/self.camera.lowerRes[0])*self.camera.start_x)
            end_width = int((self.camera.fullRes[0]/self.camera.lowerRes[0])*self.camera.end_x)
            threshold = 0.8

        # ask user to type in the name of the waypoint
        feature_name = simpledialog.askstring("Input", "Enter Feature Name:", parent=self)
        self.waypoint.Waypoint_table.insert("", 'end', iid=str(len(self.memory)), values=(feature_name,'Unchecked', str(self.movement.x)+" "+str(self.movement.y)+" "+str(self.movement.z)+" "+str(self.movement.rotate)))

        # Save the point into the memory list
        point = [self.movement.x, self.movement.y, self.movement.z,
                 self.movement.rotate, template_gray, threshold, focus_length, start_height, end_height, start_width, end_width,
                 feature_name]
        self.memory.append(point)

        self.camera.camera.set_controls({"AfMode": 2, "AfSpeed": 1})

        self.camera.snap = False
        self.camera.liveview.delete("roi")
        self.camera.start_x = None
        self.camera.start_y = None
        self.camera.end_x = None
        self.camera.end_y = None

        self.Result_Label.configure(foreground='green')
        self.inspection_result.set('Saved')

    def train_point(self):
        # Manual teaching has done
        if len(self.memory) > 0:
            self.train_flag = True

    def start_inspection(self):
        self.inspect_flag = True

    def pause_inspection(self):
        self.inspect_flag = False
        self.train_flag = False

    def generate_adaptive_threshold(self):
        self.Result_Label.configure(foreground='blue')
        self.inspection_result.set('Calculating')

        for i in range(len(self.memory)):
            if self.memory[i][4]:
                print("Length of images: "+ str(len(self.memory[i][4])))
                scores = []
                n = 0
                while n < len(self.memory[i][4])-1:
                    l = n+1
                    while l < len(self.memory[i][4]):
                        result = cv2.matchTemplate(self.memory[i][4][l], self.memory[i][4][n], cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, _ = cv2.minMaxLoc(result)
                        print("max value of "+self.memory[i][11] + ": " + str(max_val))
                        scores.append(max_val)
                        l += 1
                    n += 1
                mean = np.mean(scores)
                std = np.std(scores)
                self.memory[i][5] = mean - 3*std
                print(" ")
                print("mean of "+self.memory[i][11] + ": " + str(self.memory[i][5]))
                print(" ")
        self.Result_Label.configure(foreground='green')
        self.inspection_result.set('Generated')

    def test(self):
        while not self.Inspect_mode_off:
            if self.inspect_flag or self.train_flag:   # while in inspection mode
                if len(self.memory) < 1:                        # No point to inspect
                    self.inspection_result.set("No Data")
                    self.inspect_flag = self.train_flag = False
                    print("Inspection/Training completed")
                elif self.memory_index == len(self.memory):     # Has inspected all the points
                    
                    if self.train_flag:                
                        self.Result_Label.configure(foreground='green')
                        self.inspection_result.set('Image saved')

                    else:
                        self.Result_Label.configure(foreground='green')
                        self.inspection_result.set('Inspection Finished')
                        print('Inspection Finished')
                    self.memory_index = 0
                    self.inspect_flag = self.train_flag = False

                elif self.memory_index < len(self.memory):      # Inspecting

                    self.Result_Label.configure(foreground='blue')
                    self.inspection_result.set('Moving')
                    curr_point = self.memory[self.memory_index]
                    x = curr_point[0]
                    y = curr_point[1]
                    z = curr_point[2]
                    rotate = curr_point[3]
                    templates = curr_point[4]
                    threshold = curr_point[5]
                    focus_length = curr_point[6]
                    start_height = curr_point[7]
                    end_height = curr_point[8]
                    start_width = curr_point[9]
                    end_width = curr_point[10]
                    feature_name = curr_point[11]

                    gcode = "G0 E" + str(rotate)
                    self.movement.mega.write((gcode + "\n").encode('utf-8'))
                    self.movement.mega.read_until("ok").decode('utf-8').rstrip()
                    self.movement.mega.write(("M400" + "\n").encode('utf-8'))
                    time.sleep(0.01)
                    # if self.movement.mega.in_waiting == 0:
                    self.movement.mega.write(("M114" + "\n").encode('utf-8'))

                    self.movement.wait_until_movement()

                    # time.sleep(0.01)
                    # response = self.movement.mega.readline().decode('utf-8').rstrip()
                    # while len(response) == 0 or response[0] != "X":
                    #     response = self.movement.mega.readline().decode('utf-8').rstrip()

                    gcode = "G0 X" + str(x) + " Y" + str(y) + " Z" + str(z)
                    self.movement.mega.write((gcode + "\n").encode('utf-8'))
                    self.movement.mega.read_until("ok").decode('utf-8').rstrip()
                    self.movement.mega.write(("M400" + "\n").encode('utf-8'))
                    # time.sleep(0.01)
                    # if self.movement.mega.in_waiting == 0:
                    self.movement.mega.write(("M114" + "\n").encode('utf-8'))
                    self.movement.wait_until_movement()
                    # time.sleep(0.01)
                    # response = self.movement.mega.readline().decode('utf-8').rstrip()
                    # while len(response) == 0 or response[0] != "X":
                    #     response = self.movement.mega.readline().decode('utf-8').rstrip()

                    time.sleep(0.2)
                    if focus_length is None:    # Way point
                        self.Result_Label.configure(foreground='black')
                        self.inspection_result.set('Arrived')
                        self.waypoint.Waypoint_table.item(str(self.memory_index), values=(feature_name, 'Arrived',\
                                            str(self.memory[self.memory_index][0])+" "+str(self.memory[self.memory_index][1])+" "+str(self.memory[self.memory_index][2])+" "+str(self.memory[self.memory_index][3])))
                    else:                   # Feature point
                        # Use saved focus length and ROI to capture a target image
                        self.camera.camera.set_controls({"AfMode": 0, "LensPosition": focus_length})
                        time.sleep(0.5)
                        current_img = self.camera.camera.capture_array("main")
                        if self.train_flag:
                            cropped_current_img = current_img[(start_height - 50):(end_height + 50),
                                            (start_width - 50):(end_width + 50)]
                        else:
                            cropped_current_img = current_img[(start_height - 60):(end_height + 60),
                                                  (start_width - 60):(end_width + 60)]
                        gray_inspect_image = cv2.cvtColor(cropped_current_img, cv2.COLOR_RGB2GRAY)
                        gray_inspect_image = cv2.GaussianBlur(gray_inspect_image, (5,5), 0)

                        # If automatically training, save template image to the second deque
                        if self.train_flag:
                            
                            if len(templates) == 0:             # After the manual teaching, when no image is saved.
                                self.memory[self.memory_index][4].append(gray_inspect_image)
                                img= Image.fromarray(gray_inspect_image)
                                file_name = "Template"+ str(len(self.memory[self.memory_index][4])) + " "+str(self.memory_index)+".jpeg"
                                img.save(file_name)
                            else:
                                result = cv2.matchTemplate(gray_inspect_image, templates[0], cv2.TM_CCOEFF_NORMED)
                                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                                top_left = max_loc
                                bottom_right = (top_left[0]+((end_width+50) - (start_width-50)), top_left[1]+((end_height+50)-(start_height-50)))
                                gray_inspect_image = gray_inspect_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                                self.memory[self.memory_index][4].append(gray_inspect_image)
                                img= Image.fromarray(gray_inspect_image)
                                file_name = "Training"+ str(len(self.memory[self.memory_index][4])) + " "+str(self.memory_index)+".jpeg"
                                img.save(file_name)

                        else:
                            # Compare reference and target image by template matching
                            mean = total = 0
                            for template in templates:
                                result = cv2.matchTemplate(gray_inspect_image, template, cv2.TM_CCOEFF_NORMED)
                                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                                total = max_val + total
                            mean = total/len(templates)

                            # Show the result of inspection in the GUI
                            print("max_val: ", mean)

                            if mean >= threshold:
                                self.Result_Label.configure(foreground='green')
                                self.inspection_result.set('Pass')
                                self.waypoint.Waypoint_table.item(str(self.memory_index), values=(feature_name, 'Pass',\
                                                    str(self.memory[self.memory_index][0])+" "+str(self.memory[self.memory_index][1])+\
                                                        " "+str(self.memory[self.memory_index][2])+" "+str(self.memory[self.memory_index][3])))
                            else:
                                self.Result_Label.configure(foreground='red')
                                self.inspection_result.set('Fail')
                                self.waypoint.Waypoint_table.item(str(self.memory_index), values=(feature_name, 'Fail',\
                                                    str(self.memory[self.memory_index][0])+" "+str(self.memory[self.memory_index][1])\
                                                        +" "+str(self.memory[self.memory_index][2])+" "+str(self.memory[self.memory_index][3])))
                            img= Image.fromarray(gray_inspect_image)
                            file_name = "Inspecting"+str(self.memory_index)+".jpeg"
                            img.save(file_name)
                    # Go to next point
 
                    self.memory_index += 1

            else:   # while pause the inspection
                time.sleep(1)

    # Load a memory list of points from a .pickle file and display the in the waypoint table of the GUI
    def OpenFile(self):
        file_path = filedialog.askopenfilename(initialdir="home\\pi\\FinalProject",
                                              title="Load Memory",
                                              filetypes=[("pickle files", "*.pickle")])
        if file_path:
            with open(file_path, 'rb') as file:
                self.NewFile()
                self.memory = pickle.load(file)

                for i in range(len(self.memory)):
                    self.waypoint.Waypoint_table.insert("", 'end', iid=i,
                                                        values=(self.memory[i][11], 'Unchecked', str(self.memory[i][0])+" "+str(self.memory[i][1])+" "+str(self.memory[i][2])+" "+str(self.memory[i][3])))

    # Save the memory list of points into a .pickle file
    def SaveFile(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".pickle",
                                               initialdir="home\\pi\\FinalProject",
                                               title="Saving Memory",
                                               filetypes=[("Pickle Files", "*.pickle")])
        if file_path:
            with open(file_path, 'wb') as file:
                pickle.dump(self.memory, file)

    def NewFile(self):
        for i in self.waypoint.Waypoint_table.get_children():
            self.waypoint.Waypoint_table.delete(i)
        self.memory = deque()
        self.memory_index = 0
        self.feature_point_index = 0
        self.Inspect_mode_off = False
        self.inspect_flag = False
        self.train_flag = False
        self.movement.x = self.movement.y = self.movement.z = self.movement.rotate = 0

        self.movement.mega.write(("G92 E0 X0 Y0 Z0" + "\n").encode('utf-8'))
        self.movement.mega.read_until("ok").decode('utf-8').rstrip()


class Waypoint:
    def __init__(self, window):
        self.window = window
        # Initialise
        self.frame = ttk.Frame(window)
        self.label = ttk.Label(self.frame, text="Waypoint Table", font='Calibri 14 bold')
        self.Waypoint_table = ttk.Treeview(self.frame,  # Table contains saved waypoints, store in waypoint frame
                                           columns=('Feature Name', 'Status', 'Position'),
                                           show='headings')
        self.Waypoint_table.heading('Feature Name', text='Feature Name', )
        self.Waypoint_table.heading('Status', text='Status')
        self.Waypoint_table.heading('Position', text='Position')

        # Pack
        self.label.pack()
        self.Waypoint_table.pack(expand=True, fill='both')
        self.frame.pack(pady=10, expand=True, fill='both')




class Movement:
    def __init__(self, window, app_instance):
        self.window = window
        # Initialise
        self.frame = ttk.Frame(window)
        self.label = ttk.Label(self.frame, text="Gantry Movement", font='Calibri 14 bold')
        self.app_instance = app_instance
        self.inspection_flag = self.app_instance.inspect_flag
        self.train_flag = self.app_instance.train_flag
        self.x = 0
        self.y = 0
        self.z = 0
        self.rotate = 0

        try:
            self.mega = serial.Serial(port="/dev/ttyACM0", baudrate=250000, bytesize=8, timeout=1, stopbits=serial.STOPBITS_ONE)
        except serial.SerialException as e:
            print(f"Line 426: Serial communication error: {e}")

        self.initialised = False
        self.reached = True
        self.homed = False              # Home all axes
        self.homed_X = False            # Home X axis
        self.homed_Y = False            # Home Y axis
        self.homed_Z = False            # Home Z axis
        self.homed_T = False            # Home turntable, T axis

        # Move 1 cm
        self.left_10_running = False
        self.right_10_running = False
        self.forward_10_running = False
        self.backward_10_running = False
        self.up_10_running = False
        self.down_10_running = False

        # Turn 5 degrees
        self.cw_5_running = False
        self.ccw_5_running = False

        # Move 5 cm
        self.left_50_running = False
        self.right_50_running = False
        self.forward_50_running = False
        self.backward_50_running = False
        self.up_50_running = False
        self.down_50_running = False

        # Turn 10 degrees
        self.cw_10_running = False
        self.ccw_10_running = False

        # Move 10 cm
        self.left_100_running = False
        self.right_100_running = False
        self.forward_100_running = False
        self.backward_100_running = False
        self.up_100_running = False
        self.down_100_running = False


        # self.send()
        self.initialise()

        # Tab
        self.axis = ttk.Notebook(self.frame)
        self.x_axis = ttk.Frame(self.axis)
        self.y_axis = ttk.Frame(self.axis)
        self.z_axis = ttk.Frame(self.axis)
        self.turnable_table = ttk.Frame(self.axis)
        # self.PTZ_axis = ttk.Frame(self.axis)

        # Add axis to tab
        self.axis.add(self.x_axis, text='x-axis')
        self.axis.add(self.y_axis, text='y-axis')
        self.axis.add(self.z_axis, text='z-axis')
        self.axis.add(self.turnable_table, text='turnable table')
        # self.axis.add(self.PTZ_axis, text='ptz-axis')

        # Buttons
        self.home_button = ttk.Button(self.frame, text = 'Home', command=self.go_homing, bootstyle='success-outline')
        self.home_x_button = ttk.Button(self.x_axis, text = 'Home X', command=self.go_homing_x, bootstyle='success-outline')
        self.home_y_button = ttk.Button(self.y_axis, text = 'Home Y', command=self.go_homing_y, bootstyle='success-outline')
        self.home_z_button = ttk.Button(self.z_axis, text = 'Home Z', command=self.go_homing_z, bootstyle='success-outline')
        self.home_t_button = ttk.Button(self.turnable_table, text = 'Home T', command=self.go_homing_t, bootstyle='success-outline')

        # 1cm buttons
        self.left_10_button = ttk.Button(self.x_axis, text='-1')
        self.right_10_button = ttk.Button(self.x_axis, text='+1')
        self.left_10_button.bind('<Button-1>', self.left_10_move)
        self.right_10_button.bind('<Button-1>', self.right_10_move)
        self.left_10_button.bind('<ButtonRelease-1>',self.left_10_stop)
        self.right_10_button.bind('<ButtonRelease-1>', self.right_10_stop)

        self.up_10_button = ttk.Button(self.z_axis, text='+1')
        self.down_10_button = ttk.Button(self.z_axis, text='-1')
        self.up_10_button.bind('<Button-1>', self.up_10_move)
        self.down_10_button.bind('<Button-1>', self.down_10_move)
        self.up_10_button.bind('<ButtonRelease-1>',self.up_10_stop)
        self.down_10_button.bind('<ButtonRelease-1>', self.down_10_stop)

        self.forward_10_button = ttk.Button(self.y_axis, text='+1')
        self.backward_10_button = ttk.Button(self.y_axis, text='-1')
        self.forward_10_button.bind('<Button-1>', self.forward_10_move)
        self.backward_10_button.bind('<Button-1>', self.backward_10_move)
        self.forward_10_button.bind('<ButtonRelease-1>',self.forward_10_stop)
        self.backward_10_button.bind('<ButtonRelease-1>', self.backward_10_stop)

        # 5 degrees buttons
        self.ccw_5_running_button = ttk.Button(self.turnable_table, text='CCW 5')
        self.cw_5_running_button = ttk.Button(self.turnable_table, text='CW 5')
        self.ccw_5_running_button.bind('<Button-1>', self.ccw_5_move)
        self.cw_5_running_button.bind('<Button-1>', self.cw_5_move)
        self.ccw_5_running_button.bind('<ButtonRelease-1>',self.ccw_5_stop)
        self.cw_5_running_button.bind('<ButtonRelease-1>', self.cw_5_stop)

        # 5cm button
        self.left_50_button = ttk.Button(self.x_axis, text='-5')
        self.right_50_button = ttk.Button(self.x_axis, text='+5')
        self.left_50_button.bind('<Button-1>', self.left_50_move)
        self.right_50_button.bind('<Button-1>', self.right_50_move)
        self.left_50_button.bind('<ButtonRelease-1>',self.left_50_stop)
        self.right_50_button.bind('<ButtonRelease-1>', self.right_50_stop)

        self.up_50_button = ttk.Button(self.z_axis, text='+5')
        self.down_50_button = ttk.Button(self.z_axis, text='-5')
        self.up_50_button.bind('<Button-1>', self.up_50_move)
        self.down_50_button.bind('<Button-1>', self.down_50_move)
        self.up_50_button.bind('<ButtonRelease-1>',self.up_50_stop)
        self.down_50_button.bind('<ButtonRelease-1>', self.down_50_stop)

        self.forward_50_button = ttk.Button(self.y_axis, text='+5')
        self.backward_50_button = ttk.Button(self.y_axis, text='-5')
        self.forward_50_button.bind('<Button-1>', self.forward_50_move)
        self.backward_50_button.bind('<Button-1>', self.backward_50_move)
        self.forward_50_button.bind('<ButtonRelease-1>',self.forward_50_stop)
        self.backward_50_button.bind('<ButtonRelease-1>', self.backward_50_stop)

        # 10 cm buttons
        self.left_100_button = ttk.Button(self.x_axis, text='-10')
        self.right_100_button = ttk.Button(self.x_axis, text='+10')
        self.left_100_button.bind('<Button-1>', self.left_100_move)
        self.right_100_button.bind('<Button-1>', self.right_100_move)
        self.left_100_button.bind('<ButtonRelease-1>',self.left_100_stop)
        self.right_100_button.bind('<ButtonRelease-1>', self.right_100_stop)

        self.up_100_button = ttk.Button(self.z_axis, text='+10')
        self.down_100_button = ttk.Button(self.z_axis, text='-10')
        self.up_100_button.bind('<Button-1>', self.up_100_move)
        self.down_100_button.bind('<Button-1>', self.down_100_move)
        self.up_100_button.bind('<ButtonRelease-1>',self.up_100_stop)
        self.down_100_button.bind('<ButtonRelease-1>', self.down_100_stop)

        self.forward_100_button = ttk.Button(self.y_axis, text='+10')
        self.backward_100_button = ttk.Button(self.y_axis, text='-10')
        self.forward_100_button.bind('<Button-1>', self.forward_100_move)
        self.backward_100_button.bind('<Button-1>', self.backward_100_move)
        self.forward_100_button.bind('<ButtonRelease-1>',self.forward_100_stop)
        self.backward_100_button.bind('<ButtonRelease-1>', self.backward_100_stop)

        # 10 degrees buttons
        self.ccw_10_running_button = ttk.Button(self.turnable_table, text='CCW 10')
        self.cw_10_running_button = ttk.Button(self.turnable_table, text='CW 10')
        self.ccw_10_running_button.bind('<Button-1>', self.ccw_10_move)
        self.cw_10_running_button.bind('<Button-1>', self.cw_10_move)
        self.ccw_10_running_button.bind('<ButtonRelease-1>',self.ccw_10_stop)
        self.cw_10_running_button.bind('<ButtonRelease-1>', self.cw_10_stop)

        # Pack
        self.label.pack()
        self.axis.pack(fill='y')
        self.left_100_button.pack(side='left', expand=True, fill='both')
        self.left_50_button.pack(side='left', expand=True, fill='both')
        self.left_10_button.pack(side='left', expand=True, fill='both')
        self.right_10_button.pack(side='left', expand=True, fill='both')
        self.right_50_button.pack(side='left', expand=True, fill='both')
        self.right_100_button.pack(side='left', expand=True, fill='both')
        self.home_x_button.pack(side='right', expand=True, fill='both')

        self.down_100_button.pack(side='left', expand=True, fill='both')
        self.down_50_button.pack(side='left', expand=True, fill='both')
        self.down_10_button.pack(side='left', expand=True, fill='both')
        self.up_10_button.pack(side='left', expand=True, fill='both')
        self.up_50_button.pack(side='left', expand=True, fill='both')
        self.up_100_button.pack(side='left', expand=True, fill='both')
        self.home_z_button.pack(side='right', expand=True, fill='both')

        self.backward_100_button.pack(side='left', expand=True, fill='both')
        self.backward_50_button.pack(side='left', expand=True, fill='both')
        self.backward_10_button.pack(side='left', expand=True, fill='both')
        self.forward_10_button.pack(side='left', expand=True, fill='both')
        self.forward_50_button.pack(side='left', expand=True, fill='both')
        self.forward_100_button.pack(side='left', expand=True, fill='both') 
        self.home_y_button.pack(side='right', expand=True, fill='both')

        self.cw_10_running_button.pack(side='left', expand=True, fill='both')
        self.cw_5_running_button.pack(side='left', expand=True, fill='both')
        self.ccw_5_running_button.pack(side='left', expand=True, fill='both')
        self.ccw_10_running_button.pack(side='left', expand=True, fill='both')
        self.home_t_button.pack(side='right', expand=True, fill='both')

        self.frame.pack(pady=10, expand=True, fill='both')
        self.home_button.pack(expand=True, fill='both')

    # Marlin send some initialisation messages, this function waits until messages are all captured.
    def initialise(self):
        while not self.initialised:
            if self.mega.in_waiting == 0:
                print("Initialised")
                self.initialised = True
            elif self.mega.readline().decode('utf-8').rstrip() == "":
                print("Initialised")
                self.initialised = True
        self.mega.write(("M205 X8.00 Y8.00 Z8.00 E10" + "\n").encode('utf-8'))

        

    def go_homing(self):
        self.homed = True

    def go_homing_x(self):
        self.homed_X = True

    def go_homing_y(self):
        self.homed_Y = True

    def go_homing_z(self):
        self.homed_Z = True

    def go_homing_t(self):
        self.homed_T = True

    def wait_until_movement(self):
        while True:
            try:
                if self.mega.in_waiting > 0:
                    response = self.mega.readline().decode('utf-8').rstrip()
                    print(response)
                    if len(response) != 0 and response[0] == "X":
                        self.reached = True
                        break
                time.sleep(0.1)

            except serial.SerialTimeoutException:
                print("Line 644 Timeout occured")
            except serial.SerialException as e:
                print(f"Line 646: Serial communication error: {e}")
            except Exception:
                print("Line 648, Something unexpected happened")


    def send(self):
        # When not inspecting 
        while not self.inspection_flag and not self.train_flag:
            if not self.reached:
                self.mega.write(("M114" + "\n").encode('utf-8'))
                self.wait_until_movement()
                
            if self.homed and self.reached:
                # When large home button is clicked.
                # self.mega.write(("M221 S150" + "\n").encode('utf-8'))            # Slow down the turntable speed to reduce inertia that may be arised by sudden stop, so it doesn't break the limit switch 
                
                self.mega.write(("G28" + "\n").encode('utf-8'))
                self.mega.write(("G0 E360" + "\n").encode('utf-8'))              # GPT says marlin executes Gcode sequentially, thus it is expected to home the gantry, then the turntable.
                self.mega.read_until("ok").decode('utf-8').rstrip()
                # self.mega.write(("M400" + "\n").encode('utf-8'))
                self.homed = False
                self.x = self.y = self.z = self.rotate = 0

            elif self.homed_X and self.reached:
                self.mega.write(("G28 X" + "\n").encode('utf-8'))
                self.mega.read_until("ok").decode('utf-8').rstrip()
                self.mega.write(("M400" + "\n").encode('utf-8'))
                self.reached = self.homed_X = False
                self.x = 0
            
            elif self.homed_Y and self.reached:
                self.mega.write(("G28 Y" + "\n").encode('utf-8'))
                self.mega.read_until("ok").decode('utf-8').rstrip()
                self.mega.write(("M400" + "\n").encode('utf-8'))
                self.reached = self.homed_Y = False
                self.y = 0

            elif self.homed_Z and self.reached:
                self.mega.write(("G28 Z" + "\n").encode('utf-8'))
                self.mega.read_until("ok").decode('utf-8').rstrip()
                self.mega.write(("M400" + "\n").encode('utf-8'))
                self.reached = self.homed_Z = False
                self.z = 0

            elif self.homed_T and self.reached:
                self.mega.write(("G0 E360" + "\n").encode('utf-8'))
                self.mega.read_until("ok").decode('utf-8').rstrip()
                self.homed_T = False
                
            elif self.left_10_running and self.reached:
                self.x = self.x - 10
                gcode = "G0 X" + str(self.x) 
                self.mega.write((gcode + "\n").encode('utf-8'))
                self.mega.read_until("ok").decode('utf-8').rstrip()
                self.mega.write(("M400" + "\n").encode('utf-8'))
                self.reached = False

            elif self.right_10_running and self.reached:
                self.x = self.x + 10
                gcode = "G0 X" + str(self.x) 
                self.mega.write((gcode + "\n").encode('utf-8'))
                self.mega.read_until("ok").decode('utf-8').rstrip()
                self.mega.write(("M400" + "\n").encode('utf-8'))
                self.reached = False

            elif self.up_10_running and self.reached:
                self.z = self.z - 10
                gcode = "G0 Z" + str(self.z)
                self.mega.write((gcode + "\n").encode('utf-8'))
                self.mega.read_until("ok").decode('utf-8').rstrip()
                self.mega.write(("M400" + "\n").encode('utf-8'))
                self.reached = False

            elif self.down_10_running and self.reached:
                self.z = self.z + 10
                gcode = "G0 Z" + str(self.z)
                self.mega.write((gcode + "\n").encode('utf-8'))
                self.mega.read_until("ok").decode('utf-8').rstrip()
                self.mega.write(("M400" + "\n").encode('utf-8'))
                self.reached = False

            elif self.forward_10_running and self.reached:
                self.y = self.y + 10
                gcode = "G0 Y" + str(self.y)
                self.mega.write((gcode + "\n").encode('utf-8'))
                self.mega.read_until("ok").decode('utf-8').rstrip()
                self.mega.write(("M400" + "\n").encode('utf-8'))
                self.reached = False

            elif self.backward_10_running and self.reached:
                self.y = self.y - 10
                gcode = "G0 Y" + str(self.y)
                self.mega.write((gcode + "\n").encode('utf-8'))
                self.mega.read_until("ok").decode('utf-8').rstrip()
                self.mega.write(("M400" + "\n").encode('utf-8'))
                self.reached = False

            elif self.cw_5_running and self.reached:
                self.rotate = self.rotate + 5
                gcode = "G0 E" + str(self.rotate)
                self.mega.write((gcode + "\n").encode('utf-8'))
                self.mega.read_until("ok").decode('utf-8').rstrip()
                self.mega.write(("M400" + "\n").encode('utf-8'))
                self.reached = False

            elif self.ccw_5_running and self.reached:
                self.rotate = self.rotate - 5
                gcode = "G0 E" + str(self.rotate)
                self.mega.write((gcode + "\n").encode('utf-8'))
                self.mega.read_until("ok").decode('utf-8').rstrip()
                self.mega.write(("M400" + "\n").encode('utf-8'))
                self.reached = False
            
            elif self.left_50_running and self.reached:
                self.x = self.x - 50
                gcode = "G0 X" + str(self.x) 
                self.mega.write((gcode + "\n").encode('utf-8'))
                self.mega.read_until("ok").decode('utf-8').rstrip()
                self.mega.write(("M400" + "\n").encode('utf-8'))
                self.reached = False

            elif self.right_50_running and self.reached:
                self.x = self.x + 50
                gcode = "G0 X" + str(self.x) 
                self.mega.write((gcode + "\n").encode('utf-8'))
                self.mega.read_until("ok").decode('utf-8').rstrip()
                self.mega.write(("M400" + "\n").encode('utf-8'))
                self.reached = False

            elif self.up_50_running and self.reached:
                self.z = self.z - 50
                gcode = "G0 Z" + str(self.z)
                self.mega.write((gcode + "\n").encode('utf-8'))
                self.mega.read_until("ok").decode('utf-8').rstrip()
                self.mega.write(("M400" + "\n").encode('utf-8'))
                self.reached = False

            elif self.down_50_running and self.reached:
                self.z = self.z + 50
                gcode = "G0 Z" + str(self.z)
                self.mega.write((gcode + "\n").encode('utf-8'))
                self.mega.read_until("ok").decode('utf-8').rstrip()
                self.mega.write(("M400" + "\n").encode('utf-8'))
                self.reached = False

            elif self.forward_50_running and self.reached:
                self.y = self.y + 50
                gcode = "G0 Y" + str(self.y)
                self.mega.write((gcode + "\n").encode('utf-8'))
                self.mega.read_until("ok").decode('utf-8').rstrip()
                self.mega.write(("M400" + "\n").encode('utf-8'))
                self.reached = False

            elif self.backward_50_running and self.reached:
                self.y = self.y - 50
                gcode = "G0 Y" + str(self.y)
                self.mega.write((gcode + "\n").encode('utf-8'))
                self.mega.read_until("ok").decode('utf-8').rstrip()
                self.mega.write(("M400" + "\n").encode('utf-8'))
                self.reached = False

            elif self.cw_10_running and self.reached:
                self.rotate = self.rotate + 10
                gcode = "G0 E" + str(self.rotate)
                self.mega.write((gcode + "\n").encode('utf-8'))
                self.mega.read_until("ok").decode('utf-8').rstrip()
                self.mega.write(("M400" + "\n").encode('utf-8'))
                self.reached = False

            elif self.ccw_10_running and self.reached:
                self.rotate = self.rotate - 10
                gcode = "G0 E" + str(self.rotate)
                self.mega.write((gcode + "\n").encode('utf-8'))
                self.mega.read_until("ok").decode('utf-8').rstrip()
                self.mega.write(("M400" + "\n").encode('utf-8'))
                self.reached = False

            elif self.left_100_running and self.reached:
                self.x = self.x - 100
                gcode = "G0 X" + str(self.x) 
                self.mega.write((gcode + "\n").encode('utf-8'))
                self.mega.read_until("ok").decode('utf-8').rstrip()
                self.mega.write(("M400" + "\n").encode('utf-8'))
                self.reached = False

            elif self.right_100_running and self.reached:
                self.x = self.x + 100
                gcode = "G0 X" + str(self.x) 
                self.mega.write((gcode + "\n").encode('utf-8'))
                self.mega.read_until("ok").decode('utf-8').rstrip()
                self.mega.write(("M400" + "\n").encode('utf-8'))
                self.reached = False

            elif self.up_100_running and self.reached:
                self.z = self.z - 100
                gcode = "G0 Z" + str(self.z)
                self.mega.write((gcode + "\n").encode('utf-8'))
                self.mega.read_until("ok").decode('utf-8').rstrip()
                self.mega.write(("M400" + "\n").encode('utf-8'))
                self.reached = False

            elif self.down_100_running and self.reached:
                self.z = self.z + 100
                gcode = "G0 Z" + str(self.z)
                self.mega.write((gcode + "\n").encode('utf-8'))
                self.mega.read_until("ok").decode('utf-8').rstrip()
                self.mega.write(("M400" + "\n").encode('utf-8'))
                self.reached = False

            elif self.forward_100_running and self.reached:
                self.y = self.y + 100
                gcode = "G0 Y" + str(self.y)
                self.mega.write((gcode + "\n").encode('utf-8'))
                self.mega.read_until("ok").decode('utf-8').rstrip()
                self.mega.write(("M400" + "\n").encode('utf-8'))
                self.reached = False

            elif self.backward_100_running and self.reached:
                self.y = self.y - 100
                gcode = "G0 Y" + str(self.y)
                self.mega.write((gcode + "\n").encode('utf-8'))
                self.mega.read_until("ok").decode('utf-8').rstrip()
                self.mega.write(("M400" + "\n").encode('utf-8'))
                self.reached = False

        print("After send")


    #-----------------------------------
    # For 10 cm
    def left_100_move(self, event):
        self.left_100_running = True

    def left_100_stop(self, event):
        self.left_100_running = False

    def right_100_move(self, event):
        self.right_100_running = True

    def right_100_stop(self, event):
        self.right_100_running = False

    def forward_100_move(self, event):
        self.forward_100_running = True

    def forward_100_stop(self, event):
        self.forward_100_running = False

    def backward_100_move(self, event):
        self.backward_100_running = True

    def backward_100_stop(self, event):
        self.backward_100_running = False

    def up_100_move(self, event):
        self.up_100_running = True

    def up_100_stop(self, event):
        self.up_100_running = False

    def down_100_move(self, event):
        self.down_100_running = True

    def down_100_stop(self, event):
        self.down_100_running = False

    # -----------------------------------------------
    # For 1 cm
    def left_10_move(self, event):
        self.left_10_running = True

    def left_10_stop(self, event):
        self.left_10_running = False

    def right_10_move(self, event):
        self.right_10_running = True

    def right_10_stop(self, event):
        self.right_10_running = False

    def forward_10_move(self, event):
        self.forward_10_running = True

    def forward_10_stop(self, event):
        self.forward_10_running = False

    def backward_10_move(self, event):
        self.backward_10_running = True

    def backward_10_stop(self, event):
        self.backward_10_running = False

    def up_10_move(self, event):
        self.up_10_running = True

    def up_10_stop(self, event):
        self.up_10_running = False

    def down_10_move(self, event):
        self.down_10_running = True

    def down_10_stop(self, event):
        self.down_10_running = False

    # --------------------------------------
    # 5 deg
    def ccw_5_move(self, event):
        self.ccw_5_running = True

    def ccw_5_stop(self, event):
        self.ccw_5_running = False

    def cw_5_move(self, event):
        self.cw_5_running = True

    def cw_5_stop(self, event):
        self.cw_5_running = False

    # --------------------------------------------
    # 5cm
    def left_50_move(self, event):
        self.left_50_running = True

    def left_50_stop(self, event):
        self.left_50_running = False

    def right_50_move(self, event):
        self.right_50_running = True

    def right_50_stop(self, event):
        self.right_50_running = False

    def forward_50_move(self, event):
        self.forward_50_running = True

    def forward_50_stop(self, event):
        self.forward_50_running = False

    def backward_50_move(self, event):
        self.backward_50_running = True

    def backward_50_stop(self, event):
        self.backward_50_running = False

    def up_50_move(self, event):
        self.up_50_running = True

    def up_50_stop(self, event):
        self.up_50_running = False

    def down_50_move(self, event):
        self.down_50_running = True

    def down_50_stop(self, event):
        self.down_50_running = False

    # ---------------------------------
    # 10 deg
    def ccw_10_move(self, event):
        self.ccw_10_running = True

    def ccw_10_stop(self, event):
        self.ccw_10_running = False

    def cw_10_move(self, event):
        self.cw_10_running = True

    def cw_10_stop(self, event):
        self.cw_10_running = False

class CameraView:
    def __init__(self, window):
        # Initialise
        self.window = window
        self.frame = ttk.Frame(self.window)
        self.label = ttk.Label(self.frame, text="Camera Liveview", font='Calibri 14 bold')

        # Initialise ROI
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None
        self.rectangle = None

        # Camera
        self.camera = Picamera2()
        self.lowerRes = [dim // 8 for dim in self.camera.camera_properties["PixelArraySize"]]
        self.fullRes = self.camera.camera_properties["PixelArraySize"]

        # Display size
        self.video_width = self.lowerRes[0]
        self.video_height = self.lowerRes[1]

        # Start Camera
        self.cam_config = self.camera.create_still_configuration(lores={"size":self.lowerRes}, transform=Transform(hflip=True, vflip=True))
        self.camera.configure(self.cam_config)
        self.camera.start()
        self.camera.set_controls({"AfMode": 2, "AfSpeed": 1})

        # Create widgets
        self.liveview = ttk.Canvas(self.frame, width=self.video_width, height=self.video_height)
        self.button_frame = ttk.Frame(self.frame)
        self.snapshot_btn = ttk.Button(self.button_frame, text='SnapShot', command=self.snapshot,
                                       bootstyle='success-outline')
        self.focus_button = ttk.Button(self.button_frame, text='Focus', command=self.re_focus,
                                       bootstyle='success-outline')
        self.snap = False

        # Pack
        self.label.pack()
        self.liveview.pack(padx=10)
        self.focus_button.pack(side='left', padx=10, pady=5)
        self.snapshot_btn.pack(side='left', padx=10, pady=5)
        self.button_frame.pack()
        self.frame.pack(padx=10)

        # Enable drawing roi
        self.liveview.bind("<Button-1>", self.on_mouse_press)
        self.liveview.bind("<B1-Motion>", self.on_mouse_motion)

        # update camera liive-view in the GUI
        self.update()

    def re_focus(self):
        self.camera.autofocus_cycle()

    def snapshot(self):
        self.snap = not self.snap

    def on_mouse_press(self, event):
        self.liveview.delete("roi")
        if not self.snap: return
        self.start_x = event.x
        self.start_y = event.y
        self.rectangle = self.liveview.create_rectangle(self.start_x,
                                                        self.start_y,
                                                        self.start_x,
                                                        self.start_y, outline='red', tags='roi')

    # Recording the ROI based on the mouse motion
    def on_mouse_motion(self, event):
        if not self.snap: return
        self.end_x = event.x
        self.end_y = event.y
        self.liveview.coords(self.rectangle, self.start_x,
                             self.start_y, self.end_x, self.end_y)

    def update(self):
        if not self.snap:
            low_stream = self.camera.capture_array('lores')
            image = cv2.cvtColor(low_stream, cv2.COLOR_YUV420p2BGRA)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(image))
            self.liveview.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.window.after(50, self.update)

if __name__ == "__main__":
    app = app()
    app.movement.mega.close()
    app.camera.camera.close()
   
