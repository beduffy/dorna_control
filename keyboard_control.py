import subprocess
import os
from pprint import pprint
import json
import time
import sys, select, termios, tty
import traceback
import json

# import rospy
# from rosserial_arduino.srv import Test
from dorna import Dorna

msg = """
Reading from the keyboard to control Dorna Arm!
---------------------------
CTRL-C to quit
"""

moveBindings = {
		# x control
		'q': (1,0,0),
		'a': (-1,0,0),
		# y control
		'w': (0,1,0),
		's': (0,-1,0),
		# z control
		'e': (0,0,1),
		'd': (0,0,-1),
	}

def getKey():
	tty.setraw(sys.stdin.fileno())
	select.select([sys.stdin], [], [], 0)
	key = sys.stdin.read(1)
	termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
	return key

def generate_command(move_cmd, movement=1, coord_sys='xyz'):
	""" 
	Using the cartesian system, X, Y, and Z represent the position of the head of the robot. A represents the angle of the 
	tooltip (j1+j2+j3) with the xy plane and B is j4
	"""
	w_pitch = 0
	w_roll = 0
	if len(move_cmd) == 3:
		x, y, z = move_cmd
	elif len(move_cmd) == 5:
		x, y, z, w_pitch, w_roll = move_cmd
		print(x, y, z, w_pitch, w_roll)

	path_type = 'joint'

	if coord_sys == 'xyz':
		# cmd =  {"command": "move", "prm": {'path': 'joint', 'movement': movement, 'xyz': [x, y, z, 0, 0]}}
		cmd =  {"command": "move", "prm": {'path': path_type, 'movement': movement, 'xyz': [x, y, z, w_pitch, w_roll], 'speed': 5000}}
		# cmd =  {"command": "move", "prm": {'path': path_type, 'movement': movement, 'xyz': [x, y, z, w_pitch, w_roll]}}
	elif coord_sys == 'joint':
		cmd = {"command": "move", "prm": {'path': path_type, 'movement': movement, 'joint': [0, 0, 0, w_pitch, w_roll]}}
	else:
		raise ValueError("Invalid coord sys!")
	return cmd

def activate_gripper(gripper_state):
	if gripper_state == 0:
		microsecond_delay = 700
	elif gripper_state == 1:
		microsecond_delay = 1000
	elif gripper_state == 2:
		microsecond_delay = 1400
	elif gripper_state == 3:
		microsecond_delay = 1600

	try:
		print('Setting gripper to {} microsecond delay'.format(microsecond_delay))
		subprocess.Popen(["./close_gripper.sh", '-m', str(microsecond_delay)])
	except Exception as e:
		print(e)

if __name__=="__main__":
	#os.system('source activate py27 && source home/beduffy/all_projects/arm_control_ros/devel/setup.bash && python /home/beduffy/all_projects/arm_control_ros/src/arm_control/control/scripts/rosservice_call_servo_gripper.py -m 700')
	# subprocess.run("bash -c '/home/beduffy/anaconda/envs/py27/bin /home/beduffy/anaconda/envs/py27 && ... && source deactivate'" shell=True)
	# subprocess.run("bash -c 'source /home/beduffy/anaconda/envs/py27/bin/activate /home/beduffy/anaconda/envs/py27'", shell=True)
	# subprocess.run("bash -c 'source /home/beduffy/anaconda/bin/activate /home/beduffy/anaconda/envs/py27' && bash -c 'source /home/beduffy/all_projects/arm_control_ros/devel/setup.bash' && python /home/beduffy/all_projects/arm_control_ros/src/arm_control/control/scripts/rosservice_call_servo_gripper.py -m 1000", shell=True)
	
	# subprocess.Popen("./close_gripper.sh", shell=True)
	# subprocess.Popen("./close_gripper.sh", '1400')
	# try:
	# 	subprocess.Popen(["./close_gripper.sh", '-m', '1400'])
	# except Exception as e:
	# 	print(e)
	# subprocess.Popen(["./close_gripper.sh", '-m', '1400'])
	# os.system("bash -c close_gripper.sh")
	# sys.exit()


	# gripper_servo_service = rospy.ServiceProxy('test_srv', Test)
	# microsecond_delay = 1000
	# gripper_servo_service(str(microsecond_delay))
	# sys.exit()
	
	gripper_state = 0
	activate_gripper(gripper_state)

	# Dorna arm initialisation
	robot = Dorna("myconfig.json")
	robot.set_unit({"length": "mm"})
	robot.set_toolhead({"x": 80})  # Set gripper tip length for IK
	mv_scale = 3
	print(robot.connect())

	trajectory_fp = 'latest_trajectory.json'
	trajectory = []

	last_time_command_ran = time.time()

	# Keyboard control loop
	settings = termios.tcgetattr(sys.stdin)
	try:
		print(msg)
		while(1):
			key = getKey()
			if key in moveBindings.keys():
				dxyz = tuple([mv_scale * param for param in moveBindings[key]])
				command = generate_command(dxyz)
				print(command)
				# todo debounce and append and stutter commands?
				# wait 1 second before running next command
				time_to_last_command = time.time() - last_time_command_ran
				# if time_to_last_command > 0.25:
				if robot._device["state"] == 0:
				# if True:
					print('In keyboard control code: commands: {}'.format(robot._system["command"]))
					# with open('result.json', 'w') as fp:
					#     json.dump(robot._system["command"], fp)
					pprint(robot._system["command"])
					robot.play(command)
					last_time_command_ran = time.time()
				else:
					print('Last command was run before 1 second ago, so not running this command')
				
				# if (status == 14):
				#     print(msg)
				# status = (status + 1) % 15
			else:
				dxyz = (0, 0, 0)

				if (key == '\x03'):
					break
				elif key == 'p': # stop robot
					print('Halting robot')
					robot.halt()
				elif key == 'o':  # check position status
					print(robot.position("xyz"))
				elif key == 'r': # try to re-connect in case of connection missing
					robot.connect()
				elif key == 'h':
					for joint in ["j3", "j2", "j1", "j0"]:
						robot.home(joint)
						print("{} homed".format(joint))
				elif key == '0':
					dxyz = [0, 0, 0]
					command = generate_command(dxyz, movement=0, coord_sys='joint')
					print(command)
					robot.play(command)
					print("Robot at 0")
				elif key == 'g':
					gripper_state += 1
					if gripper_state == 4:
						gripper_state = 0
					activate_gripper(gripper_state)
				elif key == 'c':
					robot.calibrate([0, 0, 0, 0 ,0])
					print("Robot calibrated")
				# elif key == 'l':
				# 	robot.move_circle({'movement': 1})
				# 	print("Robot circle")
				elif key == 'b':
					pos = json.loads(robot.position("xyz"))
					print("Saving position at: {}".format(pos))
					trajectory.append(pos)
				elif key == 'y':
					# trajectory_fp
					# with open('data.json', 'w') as fp:
					#     json.dump(data, fp)
					# print('Running saved trajectory at filepath: {}'.format(trajectory_fp))

					print('Running saved trajectory')

					for xyz_point in trajectory:
						# import pdb;pdb.set_trace()
						# xyz_point = xyz_point[:3]
						command = generate_command(xyz_point, movement=0)
						print(command)
						count = 0
						max_count = 1000
						# played_command = False
						if robot._device["state"] == 0:
							robot.play(command)

							while not robot._device["state"] == 0:
								time.sleep(0.01)
								count += 1
								if count >= max_count:
									print('State never become zero! breaking')
									break
							if count < max_count:
								print('Count below max, iterating to next command')
								continue
								# robot.play(command)
								# played_command = True
							print('Past while loop, robot state: ', robot._device["state"])
							# else:
							# 	# pprint(robot._system["command"])
							# 	robot.play(command)
							# 	# played_command = True
						else:
							print('Robot busy, no command ran in trajectory')
							break
				elif key == '.':
					dxyz = [0, 0, 0, 0, 5]
					command = generate_command(dxyz, movement=1, coord_sys='joint')
					if robot._device["state"] == 0:
						robot.play(command)
						print('Rolling +5')
				elif key == ',':
					dxyz = [0, 0, 0, 0, -5]
					command = generate_command(dxyz, movement=1, coord_sys='joint')
					if robot._device["state"] == 0:
						robot.play(command)
						print('Rolling -5')
				elif key == '#':
					dxyz = [0, 0, 0, 5, 0]
					command = generate_command(dxyz, movement=1, coord_sys='joint')
					if robot._device["state"] == 0:
						robot.play(command)
						print('wrist pitch +5')
				elif key == ';':
					dxyz = [0, 0, 0, -5, 0]
					command = generate_command(dxyz, movement=1, coord_sys='joint')
					if robot._device["state"] == 0:
						robot.play(command)
						print('wrist pitch -5')
				elif key == 't':
					robot.terminate()
					print("Robot terminated")
				elif key == '+':
					mv_scale += 1
					print("Increase mv_scale: {}".format(mv_scale))
				elif key == '-':
					mv_scale -= 1
					print("Decrease mv_scale: {}".format(mv_scale))
				elif key == '[':
					mv_scale -= 20
					print("Decrease mv_scale: {}".format(mv_scale))
				elif key == ']':
					mv_scale += 20
					print("Decrease mv_scale: {}".format(mv_scale))

	except Exception as e:
		print(e)
		traceback.print_exc()

	finally:
		termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
		robot.terminate()
		print("Robot terminated")