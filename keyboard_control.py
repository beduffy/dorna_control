import subprocess
import os
from pprint import pprint
import json
import time
import sys, select, termios, tty
import traceback
import json
import threading

from flask import Flask, request, render_template, jsonify
# TODO Flask or ROS???
# import rospy
from geometry_msgs.msg import Twist
# from rosserial_arduino.srv import Test
from dorna import Dorna

from lib.dorna_kinematics import f_k, i_k, check_ground_collision

def go_to_joint_angles(joint_angles):
	command = generate_command(joint_angles, movement=0, coord_sys='joint')
	# command = generate_command(joint_angles, movement=0, coord_sys='xyz')  # big collision mistake
	print(command)
	if robot._device["state"] == 0:
		robot.play(command)

		print('Robot moved to joint_angles')
	else:
		print('Robot busy, did not go to joint_angles')

def go_to_xyz(position):
	command = generate_command(position, movement=0, coord_sys='xyz')
	print(command)
	if robot._device["state"] == 0:
		robot.play(command)

		print('Robot moved to xyz position')
	else:
		print('Robot busy, did not go to xyz position')

app = Flask(__name__)

@app.route("/")
def home():
    return 'home'

@app.route("/go_to_xyz")
def go_to_xyz_route():
	x = request.args.get('x')
	y = request.args.get('y')
	z = request.args.get('z')
	wrist_pitch = request.args.get('wrist_pitch')

	print('xyz:', x, y, z, 'wrist_pitch: ', wrist_pitch)
	if x and y and z:
		# x, y, z = float(x), float(y), float(z)
		x, y, z, wrist_pitch = float(x), float(y), float(z), float(wrist_pitch)
		# TODO clever stuff if no wrist pitch given
		if z > 8:  # TODO change
			# wrist_pitch = -4.258319999999988
			# wrist_pitch = 0.0
			fifth_IK_value = 0.0
			xyz_pitch_roll = [x, y, z, wrist_pitch, fifth_IK_value]

			joint_angles = i_k(xyz_pitch_roll)
			print('joint_angles: ', joint_angles)

			if joint_angles:
				full_toolhead_fk, xyz_positions_of_all_joints = f_k(joint_angles)
				print('full_toolhead_fk: ', full_toolhead_fk)

				# TODO rename function
				if not check_ground_collision(xyz_positions_of_all_joints):
					go_to_joint_angles(joint_angles)  # TODO WTF is wrong with this?
					# pass
				else:
					print('Ground collision detected!!!')
					return 'failure'
			else:
				return 'failure'

			# go_to_xyz(xyz_pitch_roll)
			
			return 'success'
		else:
			print('Z too low!')
			return 'failure'

	return 'failure'

@app.route("/gripper")
def gripper_route():
	gripper_state = int(request.args.get('gripper_state'))
	# gripper_state = 3
	activate_gripper(gripper_state)
	return 'success'

# curl "http://localhost:8080/get_xyz_joint"
@app.route("/get_xyz_joint")
def get_xyz_joint_route():
	robot_pos = [float(x) for x in robot.position("xyz").strip('[]').split(', ')]
	robot_joints = [float(x) for x in robot.position("joint").strip('[]').split(', ')]
	print('robot_pos: ', robot_pos)
	print('robot_joint: ', robot_joints)
	
	return_dict = {'robot_pos': robot_pos, 'robot_joint_angles': robot_joints}

	return jsonify(return_dict)



msg = """
Reading from the keyboard to control Dorna Arm!
---------------------------
CTRL-C to quit
"""

armMoveBindings = {
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
	# TODO how to make this not mess up my terminal output?
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
		# cmd =  {"command": "move", "prm": {'path': path_type, 'movement': movement, 'xyz': [x, y, z, 0, w_roll], 'speed': 5000}}
		# cmd =  {"command": "move", "prm": {'path': path_type, 'movement': movement, 'xyz': [x, y, z, w_pitch, w_roll]}}
	elif coord_sys == 'joint':
		# cmd = {"command": "move", "prm": {'path': path_type, 'movement': movement, 'joint': [x, y, z, w_pitch, w_roll]}}
		cmd = {"command": "move", "prm": {'path': path_type, 'movement': movement, 'joint': [x, y, z, w_pitch, w_roll], 'speed': 5000}}
		# cmd = {"command": "move", "prm": {'path': path_type, 'movement': movement, 'joint': [0, 0, 0, w_pitch, w_roll]}}
	else:
		raise ValueError("Invalid coord sys!")
	return cmd

def activate_gripper(gripper_state):
	if gripper_state == 0:
		robot.servo(40)
	elif gripper_state == 1:
		robot.servo(200)
		# microsecond_delay = 1000
	elif gripper_state == 2:
		# microsecond_delay = 1400
		robot.servo(400)
	elif gripper_state == 3:
		robot.servo(675)

if __name__=="__main__":	
	gripper_state = 0

	# TODO no need for sudo if I change permissions
	# TODO slow grip function

	# Dorna arm initialisation
	# robot = Dorna("myconfig.json")
	robot = Dorna("myconfig.yaml")
	# robot = Dorna()
	robot.set_unit({"length": "mm"})
	# robot.set_toolhead({"x": 80})  # Set gripper tip length for IK
	# was 44.196 and then was 125 because I thought it was from the end of dorna. But no, it's from the centre of J3 joint so 175
	print('robot.toolhead():', robot.toolhead())

	# TODO with toolhead_x of 193, my measuring tape says 634 for base straight out but FK says 644.079. Can't really be the base angle. 
	mv_scale = 3
	print(robot.connect())
	# TODO could avoid password with /dev rules?

	# activate_gripper(gripper_state)

	# manipulation_analysis_pose = [-1.350630230199798, 104.99476770048315, 552.4727117125126, -26.1, 200]
	# manipulation_analysis_pose = [0.4824848511110624, 45.81260132790233, 521.629980020351, -36.100019999999994, 199.99999000000003]
	# navigation_mode_pose = [0.5243962854933291, 49.79214976349321, 399.2267074597666, -26.1, 200]
	# navigation_mode_look_down_pose = [0.35380074678137585, 176.88710773800682, 440.02462126337844, -108.02028500000002, 199.999995]

	# todo navigation_mode_look_down_pose
	pick_up_object_right_in_front_pose = [4.732975534515328, 186.6211937924129, -178.6453444895452, -61.19049000000004, 0.0]
	manipulation_analysis_pose = [-0.7070520139752532, 70.7964863471116, 537.813061901588, -1.188899999999993, 0.0]
	navigation_mode_pose = [-2.5241238884508714, 38.54357006633009, 395.20905918415366, -4.258319999999988, 0.0]
	drop_into_bin_pose = [-0.6301972873213746, 260.7045089923903, 568.3996025850317, 53.81109000000003, 0.0]

	# rospy.init_node('arm_and_wheel_control')
	# pub = rospy.Publisher('twist', Twist, queue_size=1)

	# speed = rospy.get_param("~speed", 0.25)
	# turn = rospy.get_param("~turn", 1.0)
	x = 0
	y = 0
	z = 0
	th = 0

	joint_change_amt = 10  # TODO key for changing this? and can we calibrate even more if below 1 degree?? Of course?
	joint_change_amt = 1

	last_time_command_ran = time.time()

	# app.run(host='0.0.0.0', port=8080)
	flask_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=8080, use_reloader=False))
	# threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5005)).start()
	flask_thread.setDaemon(True)
	flask_thread.start()

	# Keyboard control loop
	settings = termios.tcgetattr(sys.stdin)
	try:
		print(msg)
		while(1):
			key = getKey()			
			if key in armMoveBindings.keys():
				dxyz = tuple([mv_scale * param for param in armMoveBindings[key]])
				command = generate_command(dxyz)
				print(command)
				if robot._device["state"] == 0:
					print('In keyboard control code: commands: {}'.format(robot._system["command"]))
					pprint(robot._system["command"])
					robot.play(command)
					last_time_command_ran = time.time()
				else:
					print('Last command was run too soon, state is not 0')
			else:
				dxyz = (0, 0, 0)

				if key == 'p': # stop robot
					print('Halting robot')
					robot.halt()
				elif key == 'o':  # check position status
					if 'null' not in robot.position("xyz"):
						robot_pos = [round(float(x), 3) for x in robot.position("xyz").strip('[]').split(', ')]
						robot_joints = [round(float(x), 3) for x in robot.position("joint").strip('[]').split(', ')]
						print('robot_pos: ', robot_pos)
						print('robot_joint: ', robot_joints)
						# print([round(float(x), 3) for x in robot.position("xyz")])  # it's a string? TODO fix 
						# TODO print position after every command so I don't have to keep showing it? Also no whitespace please
					else:
						print('Null in position')
				elif key == 'r': # try to re-connect in case of connection missing
					robot.connect()
				elif key == 'h':
					for joint in ["j3", "j2", "j1", "j0"]:
						robot.home(joint)
						print("{} homed".format(joint))

					# robot.set_joint({"j4": 0}, True)
					robot.set_joint({"j4": 0, "j3": 0}, True)
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
				elif key == 'v':
					gripper_state -= 1
					if gripper_state < 0:
						gripper_state = 3
					activate_gripper(gripper_state)
				elif key == '\\':
					gripper_state = 0
					activate_gripper(gripper_state)
				elif key == 'z':
					gripper_state = 2
					activate_gripper(gripper_state)
				elif key == 'x':
					gripper_state = 3
					activate_gripper(gripper_state)
				elif key == 'c':
					robot.calibrate([0, 0, 0, 0 ,0])
					robot.save_config('myconfig.yaml')
					print("Robot calibrated, saved to myconfig.yaml")
				# elif key == '4':
				# 	robot.move_circle({'movement': 1})
				# 	print("Robot circle")
				elif key == 'b':
					pos = json.loads(robot.position("xyz"))
					print("Saving position at: {}".format(pos))
					trajectory.append(pos)
				elif key == 'y':
					print('Running saved trajectory')

					for xyz_point in trajectory:
						command = generate_command(xyz_point, movement=0)
						print(command)
						count = 0
						max_count = 1000
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
							print('Past while loop, robot state: ', robot._device["state"])
						else:
							print('Robot busy, no command ran in trajectory')
							break
				# elif key == ';':
				# 	command = generate_command(navigation_mode_look_down_pose, movement=0, coord_sys='xyz')
				# 	print(command)
				# 	if robot._device["state"] == 0:
				# 		robot.play(command)

				# 		print('Robot moved to navigation look down pose')
				# 	else:
				# 		print('Robot busy, did not go to navigation down pose')
				elif key == 'n':
					command = generate_command(navigation_mode_pose, movement=0, coord_sys='xyz')
					print(command)
					if robot._device["state"] == 0:
						robot.play(command)

						print('Robot moved to navigation pose')
					else:
						print('Robot busy, did not go to navigation pose')
				elif key == 'm':
					command = generate_command(manipulation_analysis_pose, movement=0, coord_sys='xyz')
					print(command)
					if robot._device["state"] == 0:
						robot.play(command)

						print('Robot moved to manipulation pose')
					else:
						print('Robot busy, did not go to manipulation pose')
				elif key == 'i':
					print('Inputting IK command')
					user_input = input('Enter a xy or xyz or xyz_wrist_pitch location separated by spaces\n')
					# user_input_list = [float(x) for x in user_input.split(' ')]  # TODO commas and spaces?
					user_input_list = [float(x) for x in user_input.split(', ')]
					
					# wrist_pitch = -4.258319999999988
					wrist_pitch = 0.0
					fifth_IK_value = 0.0
					move_arm = True
					if len(user_input_list) == 2:
						x, y = user_input_list
						z = 300
					elif len(user_input_list) == 3:
						x, y, z = user_input_list
					elif len(user_input_list) == 4:
						x, y, z, wrist_pitch = user_input_list
					else:
						move_arm = False

					if move_arm:
						user_inputted_3d_position = [x, y, z, wrist_pitch, fifth_IK_value]
						go_to_xyz(user_inputted_3d_position)
				elif key == 'j':
					# good battery position -52.191, 140.865, -110.363, -20.846, 0.0
					# good 0 base angle position to test if problem is base angle: 0.0, 160.0, -118.024, -53.983, 0.0
					# awkward position: -40.245, 13.872, -97.713, 112.514, 0.0
					print('Inputting joint command')
					user_input = input('Enter 5 joint angles separated by commas \n')
					# user_input_list = [float(x) for x in user_input.split(' ')]  # TODO spaces as well?
					joint_angles = [float(x) for x in user_input.split(', ')]
					go_to_joint_angles(joint_angles)
				# elif key == '7':
				# 	command = generate_command(pick_up_object_right_in_front_pose, movement=0, coord_sys='xyz')
				# 	print(command)
				# 	if robot._device["state"] == 0:
				# 		robot.play(command)

				# 		print('Robot moved to pick_up_object_right_in_front_pose')
				# 	else:
				# 		print('Robot busy, did not go to pick_up_object_right_in_front_pose')
				# elif key == '8':
				# 	command = generate_command(drop_into_bin_pose, movement=0, coord_sys='xyz')
				# 	print(command)
				# 	if robot._device["state"] == 0:
				# 		robot.play(command)

				# 		print('Robot moved to drop_into_bin_pose')
				# 	else:
				# 		print('Robot busy, did not go to drop_into_bin_pose')
				elif key == '1':
					dxyz = [-joint_change_amt, 0, 0, 0, 0]
					command = generate_command(dxyz, movement=1, coord_sys='joint')
					if robot._device["state"] == 0:
						robot.play(command)
						print('Base angle -{}'.format(joint_change_amt))
				elif key == '2':
					dxyz = [joint_change_amt, 0, 0, 0, 0]
					command = generate_command(dxyz, movement=1, coord_sys='joint')
					if robot._device["state"] == 0:
						robot.play(command)
						print('Base angle +{}'.format(joint_change_amt))
				elif key == '3':
					dxyz = [0, -joint_change_amt, 0, 0, 0]
					command = generate_command(dxyz, movement=1, coord_sys='joint')
					if robot._device["state"] == 0:
						robot.play(command)
						print('Shoulder angle -{}'.format(joint_change_amt))
				elif key == '4':
					dxyz = [0, joint_change_amt, 0, 0, 0]
					command = generate_command(dxyz, movement=1, coord_sys='joint')
					if robot._device["state"] == 0:
						robot.play(command)
						print('Shoulder angle +{}'.format(joint_change_amt))
				elif key == '5':
					dxyz = [0, 0, -joint_change_amt, 0, 0]
					command = generate_command(dxyz, movement=1, coord_sys='joint')
					if robot._device["state"] == 0:
						robot.play(command)
						print('Elbow angle -{}'.format(joint_change_amt))
				elif key == '6':
					dxyz = [0, 0, joint_change_amt, 0, 0]
					command = generate_command(dxyz, movement=1, coord_sys='joint')
					if robot._device["state"] == 0:
						robot.play(command)
						print('Elbow angle +{}'.format(joint_change_amt))
				elif key == '#':
					dxyz = [0, 0, 0, joint_change_amt, 0]
					command = generate_command(dxyz, movement=1, coord_sys='joint')
					if robot._device["state"] == 0:
						robot.play(command)
						print('wrist pitch +{}'.format(joint_change_amt))
				elif key == '\'':
					dxyz = [0, 0, 0, -joint_change_amt, 0]
					command = generate_command(dxyz, movement=1, coord_sys='joint')
					if robot._device["state"] == 0:
						robot.play(command)
						print('wrist pitch -{}'.format(joint_change_amt))
				elif key == '.':
					dxyz = [0, 0, 0, 0, -joint_change_amt]
					command = generate_command(dxyz, movement=1, coord_sys='joint')
					if robot._device["state"] == 0:
						robot.play(command)
						print('Wrist Roll -{}'.format(joint_change_amt))
				elif key == '/':
					dxyz = [0, 0, 0, 0, joint_change_amt]
					command = generate_command(dxyz, movement=1, coord_sys='joint')
					if robot._device["state"] == 0:
						robot.play(command)
						print('Wrist Roll +{}'.format(joint_change_amt))
				elif key == 't':
					robot.terminate()
					print("Robot terminated")
				elif key == '+':
					mv_scale += 1
					print("Increase mv_scale: {}".format(mv_scale))
				elif key == '-':
					mv_scale -= 1
					if mv_scale < 0:
						mv_scale = 0
					print("Decrease mv_scale: {}".format(mv_scale))
				elif key == 'u':
					if joint_change_amt == 1:
						joint_change_amt = 10
					elif joint_change_amt == 10:
						joint_change_amt = 0.1
					elif joint_change_amt == 0.1:
						joint_change_amt = 1
					
					print("Changed joint_change_amt to: {}".format(joint_change_amt))
				elif key == ']':
					mv_scale += 20
					print("Increase mv_scale: {}".format(mv_scale))
				elif key == '[':
					mv_scale -= 20
					if mv_scale < 0:
						mv_scale = 0
					print("Decrease mv_scale: {}".format(mv_scale))
				if (key == '\x03'):
					break

	except Exception as e:
		print(e)
		traceback.print_exc()

	finally:
		# twist = Twist()
		# twist.linear.x = 0; twist.linear.y = 0; twist.linear.z = 0
		# twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = 0
		# pub.publish(twist)

		# TODO origibot gripper actually goes lower than center. I need to hack inside of dorna to fix that? or add offset elsewhere? 20mm downwards relative to center!
		# TODO z command while keeping same xy as a good pre-pick pose

		termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
		robot.terminate()
		print("Robot terminated")
