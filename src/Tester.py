import json
import numpy as np
import robot_color_tracking as track

#TODO: Resolve get_statistics in evaluate_error
#TODO: Better solution for id_options
#TODO: False Positives by ID
#TODO: Better interchange of databases

# Estatisticas


class Tester(object):
    def __init__(self, methods=['colors_naive'], trackers={}, id_options=['green','blue', 'red', 'yellow']):
        self.error = {}
        self.timeOfTrack = {}
        self.positions_data = {}
        self.statistics = {}
        self.trackers = trackers
        self.id_options = id_options
        self.i = 0
        if(len(trackers)!= 0):
            self.methods = list(trackers.keys())
        else:
            self.methods = methods
            for m in methods:
                if m == 'shapes':
                    self.trackers[m] = track.GeometricTrack(binaryThreshold= 140, segmentMethod='simple')
                elif m == 'colors_naive':
                    self.trackers[m] = track.ColorTrack(img_width=500,satTolerance= 110,binaryThreshold = 60, hueTolerance = 15,kernel = ((0,1,0),(1,1,1),(0,1,0)), colors=id_options, debug=True)
                    # id_options = ['blue','green', 'red', 'yellow']
                elif m == 'shapes_colors':
                    self.trackers[m] = track.GeometricTrack(areaBounds = (200, 20000),binaryThreshold= 70, segmentMethod='multipleColors', color=['red', 'yellow', 'blue', 'green'])
                elif m == 'hough_colors':
                    self.trackers[m] = track.HoughColorTrack(img_width=500,satTolerance= 110,binaryThreshold = 80, hueTolerance = 20,param1=30, param2=10, minRadius=1, maxRadius=500, debug=True, colors=id_options)
                    # id_options = ['blue','green', 'red', 'yellow']
                elif m == 'shapes_one_color':
                    self.trackers[m] = track.GeometricTrack(areaBounds = (200, 20000),binaryThreshold= 70, segmentMethod='oneColor',color='red')
                elif m == 'achromatic':
                    self.trackers[m] = track.AchromaticTrack(d=30,img_width=500, binaryThreshold= 40, hueTolerance=15, satTolerance=0, kernel = np.ones((3,3)),colors=id_options, debug=True)
                    # id_options = ['blue', 'green', 'red', 'yellow']
                elif m == 'ArUco':
                    self.trackers[m] = track.ArucoTrack(img_width=1000)
                else:
                    raise Exception(m+' is not recognized as a method of tracking')


        print("Tester was initialized")
    def load_positions(self):
        for m in self.methods:
            with open('../algorithm_performance_data/real_positions/'+m+'_positions.json', 'r') as f:
                self.positions_data[m] = json.load(f)
        print("Real positions loaded")
    def load_error(self, folder=''):
        for m in self.methods:
            with open('../algorithm_performance_data/real_positions/'+m+'_positions.json', 'r') as f:
                self.positions_data[m] = json.load(f)
            with open('../algorithm_performance_data/error/error_'+folder+m+'.json', 'r') as f:
                self.error[m] = json.load(f)
            with open('../algorithm_performance_data/time/time_'+folder+m+'.json', 'r') as f:
                self.timeOfTrack[m] = json.load(f)

        print("Real positions, error and time of track loaded")
    def evaluate_error(self):
        # Compara o erro entre o resultado esperado e o erro encontrado
        for m in self.methods:
            data = self.positions_data[m]
            if(m == 'ArUco'):
                for image_name in data:
                    for marker_id in data[image_name]['position']:
                        data[image_name]['position'][marker_id] = [np.array(data[image_name]['position'][marker_id]).mean(axis=0)]
            # print(data)
            # break
            path = '../img/'+m+'/'
            counterProgress = 0
            self.error[m] = {}
            for image in data:
                print('\r'+m+': '+str(format(100*counterProgress/len(data),'.2f'))+'%    i: '+str(self.i), end='')
                self.error[m][image] = { 'false_negative_total': 0,
                        'false_positive_total': 0,
                        'false_negative': {},
                        'false_positive': {},
                        'position_error': {},
                        'angle_error': {},
                        'place': data[image]['place'],
                        'tags': data[image]['tags'],
                }
                # Track and store actual and calculated poses
                self.trackers[m].track(path+image)
                # print(image)
                calc_poses = self.trackers[m].getPoses()
                # print('calc_poses: '+ str(calc_poses))
                # if(m == 'ArUco'):
                #     actual_poses = {}
                #     for marker in data[image]["position"]:
                #         actual_poses[marker] = np.array([np.array(data[image]["position"][marker]).mean(0)])
                # else:
                actual_poses = data[image]["position"]
                # print('actual_poses: '+str(actual_poses))
                # print('calc_poses: '+ str(calc_poses))
                # break
                # if m == 'achromatic' or 'colors_naive' or 'hough_colors':
                #     calc_poses['green'] = calc_poses['cyan']
                #     del(calc_poses['cyan'])
                
                for id_name in self.id_options:
                    # print('\n******'+str(actual_poses))
                    # print('\n------'+str(calc_poses))
                    if id_name in actual_poses:
                        if id_name in calc_poses:
                            err, false_positive, false_negative = self.calc_dist_error(calc_poses[id_name], actual_poses[id_name])
                            if(id_name in self.trackers[m].angle and len(actual_poses[id_name])==3):
                                angle_err = self.calc_angle_error(self.trackers[m].angle[id_name], self.trackers[m].getAngle(actual_poses[id_name]))
                            else:
                                angle_err = None
                            # print('\n ------- \n fp: '+str(false_positive)+'  \n')
                            self.error[m][image]['position_error'][id_name] = err
                            self.error[m][image]['angle_error'][id_name] = angle_err
                            self.error[m][image]['false_positive_total'] += false_positive
                            self.error[m][image]['false_positive'][id_name] = false_positive
                            self.error[m][image]['false_negative_total'] += false_negative
                            self.error[m][image]['false_negative'][id_name] = false_negative
                        else:
                            self.error[m][image]['false_negative_total']+= len(actual_poses[id_name])  
                    elif id_name in calc_poses:
                        self.error[m][image]['false_positive_total']+= len(calc_poses[id_name])
                    # print('\n ------- \n self.fp: '+str(self.error[m][image]['false_positive_total'])+'  \n')
                counterProgress+=1
                # break
            print('\r'+m+': '+format(100*counterProgress/len(data),'.2f')+'%    i: '+str(self.i))
            
            self.timeOfTrack[m] = self.trackers[m].time
    def save_error_and_time(self):
        for m in self.methods:
            with open('../algorithm_performance_data/error/error_'+m+'.json', 'w') as f:
                json.dump(self.error[m], f)
            with open('../algorithm_performance_data/time/time_'+m+'.json', 'w') as f:
                json.dump(self.timeOfTrack[m], f)

        print("Error measures and time measures are saved")

    def calc_dist_error(self,calculated_positions, actual_positions):
        actual = np.array(actual_positions)
        calculated = np.array(calculated_positions)
        result = []
        # print('\n----------------')
        # print('\nactual: ' + str(actual))
        # print('\ncalculated: ' + str(calculated))
        
        for i in range(actual.size):
            if actual.size > 0 and calculated.size > 0:
                
                dist = calculated-actual[0]
                # print(dist)
                # print(dist[:,1])
                dist = np.sqrt(dist[:,0]**2+dist[:,1]**2)
                
                # print(dist.min())
                if(dist.min()<100):
                    result.append(dist.min())
                    calculated = np.delete(calculated, dist.argmin(), 0)
                else:
                    self.i+=1
                actual = np.delete(actual, 0, 0)
            else:
                break
        
        false_positives = calculated.shape[0]
        false_negatives = actual.shape[0]
        # print('\nfalse_positives: ' + str(false_positives))
        # print('\nfalse_negatives: ' + str(false_negatives))
        # print('\ndist: ' + str(dist))
       

        return result, false_positives, false_negatives


    def calc_angle_error(self, calc_angle, actual_angle):
        return np.abs(calc_angle - actual_angle)
    def get_statistics_by_place(self):
        place_names = ["daylight", "night right bulb", "night center bulb", "night left bulb"]

        for m in self.methods:
            self.statistics[m] = {}
            self.statistics[m]['time'] = np.array(self.timeOfTrack[m]).mean()
            for place in place_names:
                self.statistics[m][place] = {}
                place_imgs = self.filter_place(self.error[m], place)
                real_positions_filtered = self.filter_place(self.positions_data[m], place)
                positions, angles = self.get_only_error(place_imgs)
                positions = np.array(positions)
                angles = np.array(angles)

                total = self.statistics[m][place]['total_positions'] = self.get_total_positions(real_positions_filtered)
                self.statistics[m][place]['distance_mean'] = positions.mean()
                self.statistics[m][place]['angle_mean'] = angles.mean()
                self.statistics[m][place]['distance_deviation'] = positions.std()
                self.statistics[m][place]['detected'] = 100*positions.size/total
                self.statistics[m][place]['total_img'] = len(place_imgs)
                self.statistics[m][place]['false_negative'] = 100*self.get_false_negative(place_imgs)/total
                self.statistics[m][place]['false_positive'] = 100*self.get_false_positive(place_imgs)/total

        print("Statistics calculated")

        return self.statistics
    def get_statistics_by_id(self, pos_data = None, err = None):
        if(pos_data == None):
            pos_data = self.positions_data
        if(err == None):
            err = self.error
        for m in self.methods:
            self.statistics[m] = {}
            self.statistics[m]['time'] = np.array(self.timeOfTrack[m]).mean()
            for robot_id in self.id_options:
                self.statistics[m][robot_id] = {}
                id_imgs = self.filter_id(err[m], robot_id)
                real_positions_filtered = self.filter_id(pos_data[m], robot_id)
                positions, angles = self.get_only_error_id(id_imgs,robot_id)
                positions = np.array(positions)
                angles = np.array(angles)

                total = self.statistics[m][robot_id]['total_positions'] = self.get_total_positions_id(real_positions_filtered, robot_id)
                self.statistics[m][robot_id]['distance_mean'] = positions.mean()
                self.statistics[m][robot_id]['angle_mean'] = angles.mean()
                self.statistics[m][robot_id]['distance_deviation'] = positions.std()
                self.statistics[m][robot_id]['detected'] = 100*positions.size/total
                self.statistics[m][robot_id]['total_img'] = len(id_imgs)
                self.statistics[m][robot_id]['false_negative'] = 100*self.get_false_negative_id(id_imgs, real_positions_filtered, robot_id)/total
                self.statistics[m][robot_id]['false_positive'] = 100*self.get_false_positive_id(id_imgs, robot_id)/total

        print("Statistics calculated")

        return self.statistics
    def filter_id(self, data, robot_id):
        filtered_data = {}
        for img in data:
            # if(data[img]['place']!='night right bulb' and data[img]['place']!='night left bulb' ):
            if('position_error' in data[img]):
                if robot_id in data[img]['position_error']:
                    filtered_data[img] = data[img]
            else:
                if robot_id in data[img]['position']:
                    filtered_data[img] = data[img]

        return filtered_data
    def filter_place(self,data, place):
        filtered_data = {}
        for img in data:
            if data[img]['place'] == place:
                filtered_data[img] = data[img]
        
        return filtered_data
    def filter_place_methods(self,data, place):
        filtered_data = {}
        for m in self.methods:
            filtered_data[m] = {}
            for img in data[m]:
                if data[m][img]['place'] == place:
                    filtered_data[m][img] = data[m][img]
        
        return filtered_data
    def get_only_error_id(self,filtered_id, robot_id):
        error_vector = []
        ang_error_vector = []
        
        for img_name in filtered_id:
            error_vector = error_vector + filtered_id[img_name]['position_error'][robot_id]
            if(filtered_id[img_name]['angle_error'][robot_id] != None):
                ang_error_vector = ang_error_vector + [filtered_id[img_name]['angle_error'][robot_id]]

        return error_vector, ang_error_vector
    def get_only_error(self,filtered_local):
        error_vector = []
        ang_error_vector = []

        for img_name in filtered_local:
            for id_name in filtered_local[img_name]['position_error']:
                error_vector = error_vector + filtered_local[img_name]['position_error'][id_name]
                try:
                    if(filtered_local[img_name]['angle_error'][robot_id] != None):
                        ang_error_vector = ang_error_vector + [filtered_local[img_name]['angle_error'][robot_id]]
                except:
                    pass
        
        return error_vector, ang_error_vector
    def get_total_positions(self,filtered_local):
        total = 0
        
        for img_name in filtered_local:
            for id_name in self.id_options:
                if(id_name in filtered_local[img_name]['position']):
                    total+= len(filtered_local[img_name]['position'][id_name])
        
        return total
    def get_total_positions_id(self,filtered_id, robot_id):
        total = 0
        
        for img_name in filtered_id:
            total+= len(filtered_id[img_name]['position'][robot_id])
        
        return total
    def get_false_positive(self,filtered_data):
        false_positive = 0
        for img_name in filtered_data:
            false_positive+= filtered_data[img_name]['false_positive_total']
        
        return false_positive
    def get_false_positive_id(self,filtered_data, robot_id):
        false_positive = 0
        for img_name in filtered_data:
            #print(filtered_data[img_name]['false_positive'][robot_id])
            false_positive+= filtered_data[img_name]['false_positive'][robot_id]
        
        return false_positive
    def get_false_negative(self,filtered_data):
        false_negative = 0
        for img_name in filtered_data:
            false_negative+= filtered_data[img_name]['false_negative_total']
        
        return false_negative
    def get_false_negative_id(self,filtered_data,real_positions, robot_id):
        false_negative = 0
        for img_name in filtered_data:
            false_negative+= len(real_positions[img_name]['position'][robot_id]) - len(filtered_data[img_name]['position_error'][robot_id])
        
        return false_negative