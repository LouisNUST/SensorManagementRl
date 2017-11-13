from target import target
from sensor import sensor
from measurement import measurement
import numpy as np
import random
import sys
from scenario import scenario
from scipy.stats import norm
import matplotlib.pyplot as plt
import sklearn.pipeline
from sklearn.kernel_approximation import RBFSampler



class EKF_tracker:
    def __init__(self,init_estimate,init_covariance,A,B,x_var,y_var,bearing_var):

        self.init_estimate = init_estimate
        self.init_covariance = init_covariance
        self.bearing_var = bearing_var
        self.A = A
        self.B = B
        self.x_var = x_var
        self.y_var = y_var

        self.x_k_k = np.array(init_estimate).reshape(len(init_estimate),1)
        self.x_k_km1 = self.x_k_k
        self.p_k_k = init_covariance
        self.p_k_km1 = init_covariance
        self.S_k = 1E-5
        self.meas_vec = []

        self.innovation_list = []
        self.innovation_var = []
        self.gain = []


    def get_linearized_measurment_vector(self,target_state,sensor_state):
        relative_location = target_state[0:2] - np.array(sensor_state[0:2]).reshape(2,1)  ##[x-x_s,y-y_s]
        measurement_vector = np.array([-relative_location[1] / ((np.linalg.norm(relative_location)) ** 2),
                                       relative_location[0] / ((np.linalg.norm(relative_location)) ** 2), [0], [0]])
        measurement_vector = measurement_vector.transpose()
        return (measurement_vector)

    def linearized_predicted_measurement(self,sensor_state):
        sensor_state = np.array(sensor_state).reshape(len(sensor_state),1)
        measurement_vector = self.get_linearized_measurment_vector(self.x_k_km1,sensor_state)#Linearize the measurement model
        #predicted_measurement = measurement_vector.dot(np.array(self.x_k_km1))
        predicted_measurement =  np.arctan2(self.x_k_km1[1]-sensor_state[1],self.x_k_km1[0]-sensor_state[0])
        if predicted_measurement<0:predicted_measurement+= 2*np.pi
        return (predicted_measurement,measurement_vector)

    def predicted_state(self,sensor_state,measurement):

        Q = np.eye(2)
        Q[0,0] = .1
        Q[1,1] = .1

        #Q[0,0] = 5
        #Q[1,1] = 5
        predicted_noise_covariance = (self.B.dot(Q)).dot(self.B.transpose())
        self.x_k_km1 = self.A.dot(self.x_k_k)
        self.p_k_km1 = (self.A.dot(self.p_k_k)).dot(self.A.transpose()) + predicted_noise_covariance
        predicted_measurement, measurement_vector = self.linearized_predicted_measurement(sensor_state)
        self.meas_vec.append(measurement_vector)
        #measurement_vector = measurement_vector.reshape(1,len(measurement_vector))
        self.S_k = (measurement_vector.dot(self.p_k_km1)).dot(measurement_vector.transpose()) + self.bearing_var
        self.innovation_list.append(measurement - predicted_measurement)
        self.innovation_var.append(self.S_k)


    def update_states(self,sensor_state,measurement):
        self.predicted_state(sensor_state,measurement)#prediction-phase
        measurement_vector = self.get_linearized_measurment_vector(self.x_k_km1,sensor_state)  # Linearize the measurement model
        #calculate Kalman gain
        kalman_gain = (self.p_k_km1.dot(measurement_vector.transpose()))/self.S_k

        self.x_k_k = self.x_k_km1 + kalman_gain*self.innovation_list[-1]
        self.p_k_k = self.p_k_km1 - (kalman_gain.dot(measurement_vector)).dot(self.p_k_km1)
        self.gain.append(kalman_gain)


def gen_learning_rate(iteration,l_max,l_min,N_max):
    if iteration>N_max: return (l_min)
    alpha = 2*l_max
    beta = np.log((alpha/l_min-1))/N_max
    return (alpha/(1+np.exp(beta*iteration)))

def normalize_state(current_state,scen):
    pos_coeff_x = np.tan(np.pi/2*.9)/max(abs(scen.x_max),abs(scen.x_min))
    pos_coeff_y = np.tan(np.pi / 2 * .9) / max(abs(scen.y_max), abs(scen.y_min))
    pos_coeff_vel = np.tan(np.pi / 2 * .9) / max(abs(scen.vel_max), abs(scen.vel_min))

    current_state[0] = 2.0/np.pi*np.arctan(pos_coeff_x*current_state[0])
    current_state[1] = 2.0 / np.pi * np.arctan(pos_coeff_y * current_state[1])
    current_state[2] = 2.0 / np.pi * np.arctan(pos_coeff_vel * current_state[2])
    current_state[3] = 2.0 / np.pi * np.arctan(pos_coeff_vel * current_state[3])
    current_state[4] = 2.0 / np.pi * np.arctan(pos_coeff_x * current_state[4])
    current_state[5] = 2.0 / np.pi * np.arctan(pos_coeff_y * current_state[5])

    return (current_state)

if __name__=="__main__":

    base_path = "/Users/u6042446/Desktop/ali_files/westlaw/DeepSensorManagement/Test/" #change this one (this is the base-path to store all required metrics)
    #Inputs:
    #1) number of RBF components (number of features)
    #2) variance of RBF kernels
    #3) initial learning rate
    #4) additive variance for exploration


    #args = sys.argv()
    #params = args[1:]
    rbf_comp = 20
    rbf_var = 1
    init_rate = .001
    add_variance = 1


    #Open writers
    post_fix = "_"+str(rbf_comp)+"_"+str(rbf_var)+"_"+str(add_variance)+"_"+str(init_rate)+".txt"
    writer_avg_reward = open(base_path+"avg_reward"+post_fix,"w")
    writer_var_reward = open(base_path+"var_reward"+post_fix,"w")
    writer_weight_update = open(base_path + "weight_update" + post_fix, "w")
    writer_final_weight = open(base_path + "final_weight" + post_fix, "w")


    RBF_COMPONENTS = rbf_comp
    #Read all randomly-generated samples for feature-generation (this is done for matching and feature construction)
    list_of_states = []
    with open("sampled_states","r") as f:
        for line in f:
            data = line.strip().split("\t")
            dd = []
            [dd.append(float(x)) for x in data]
            list_of_states.append(dd)

    featurizer = sklearn.pipeline.FeatureUnion([("rbf1", RBFSampler(gamma=rbf_var, n_components=rbf_comp))])
    featurizer.fit(np.array(list_of_states)) #Use this featurizer for normalization


    MAX_UNCERTAINTY = 1E9 #Initial uncertainty for EKF tracker
    num_states = RBF_COMPONENTS
    weight = np.random.normal(0, 1, [2, RBF_COMPONENTS]) #initialize weight matrix
    sigma_max = add_variance
    num_episodes = []
    gamma = .99 #discount factor

    episode_length = 2000 #length of each episode
    learning_rate = init_rate
    min_learning_rate = 1E-8
    N_max = 50000 #maximum number of simulations
    #parameters to calculate uncertainty
    window_size = 50 #uncertainty is calculated over a window of "window_size" iterations
    window_lag = 10
    return_saver = []
    episode_counter = 0
    weight_saver1 = []
    weight_saver2 = []
    avg_reward = []
    var_reward = []
    list_of_states = []

    #Main loop to count number of valid episodes
    while episode_counter<N_max:
        #variable variance?
        #sigma = gen_learning_rate(episode_counter,sigma_max,.1,20000)
        sigma = sigma_max
        discounted_return = np.array([])
        discount_vector = np.array([])
        #print(episodes_counter)
        scen = scenario(1,1) #create scenario object (single-target, single-sensor)
        bearing_var = 1E-2#variance of bearing measurement
        #Initialize target location + velocity randomly
        x = 2000*random.random()-1000#initial x-location
        y = 2000 * random.random() - 1000#initial y-location
        xdot = 10*random.random()-5#initial xdot-value
        ydot = 10 * random.random() - 5#initial ydot-value

        init_target_state = [x,y,xdot,ydot]#initialize target state
        #Add noise to initial target location since the tracker doesn't know about the initial location
        init_for_smc = [x+np.random.normal(0,5),y+np.random.normal(0,5),np.random.normal(0,5),np.random.normal(0,5)]#init state for the tracker (tracker doesn't know about the initial state)
        #initialize sensor location randomly too
        init_sensor_state = [2000*random.random()-1000,2000 * random.random() - 1000,3,-2]#initial sensor-state
        temp_loc = np.array(init_target_state[0:2]).reshape(2,1)
        init_location_estimate = temp_loc+0*np.random.normal(np.zeros([2,1]),10)
        init_location_estimate = [init_location_estimate[0][0],init_location_estimate[1][0]]
        init_velocity_estimate = [6*random.random()-3,6*random.random()-3]
        init_velocity_estimate = [init_target_state[2],init_target_state[3]]
        init_estimate = init_location_estimate+init_velocity_estimate
        #Initial covariance of estimation is based on MAX_UNCERTAINTY
        init_covariance = np.diag([MAX_UNCERTAINTY,MAX_UNCERTAINTY,MAX_UNCERTAINTY,MAX_UNCERTAINTY])#initial covariance of state estimation
        #Create target object
        t = target(init_target_state[0:2], init_target_state[2], init_target_state[3], 0, 0, "CONS_V")#constant-velocity model for target motion
        A, B = t.constant_velocity(1E-10)#Get motion model
        x_var = t.x_var
        y_var = t.y_var
        #create tracker object (EKF tracker)
        tracker_object = EKF_tracker(init_for_smc, init_covariance, A,B,x_var,y_var,bearing_var)#create tracker object
        #Sensor object with Stochastic Policy
        s = sensor("POLICY_COMM")#create sensor object (stochastic policy)
        #Measurement object (to generate bearing measurements)
        measure = measurement(bearing_var)#create measurement object
        #Some variables to keep important stuff
        m = []
        x_est = []; y_est = []; x_vel_est = []; y_vel_est = []
        x_truth = [];
        y_truth = [];
        x_vel_truth = [];
        y_vel_truth = []
        uncertainty = []
        vel_error = []
        pos_error = []
        iteration = []
        innovation = []
        reward = []
        episode_condition = True
        n=0
        violation = 0
        #store required information
        episode_state = []
        episode_actions = []

        #Loop over each single episode
        while episode_condition:

            #update location of target, new measurement and state of tracker
            t.update_location()
            m.append(measure.generate_bearing(t.current_location,s.current_location))
            tracker_object.update_states(s.current_location, m[-1])
            normalized_innovation = (tracker_object.innovation_list[-1])/tracker_object.innovation_var[-1]
            #Form current state s(t): target_state + sensor_state + bearing measurement + range
            current_state = list(tracker_object.x_k_k.reshape(len(tracker_object.x_k_k))) + list(s.current_location) + [m[-1]] + [np.linalg.norm(s.current_location-tracker_object.x_k_k[0:2])]
            max_node = np.array([scen.x_max, scen.y_max])
            min_node = np.array([scen.x_min, scen.y_min])
            max_distance = np.linalg.norm(max_node - min_node)
            x_slope = 2.0/(scen.x_max-scen.x_min)
            y_slope = 2.0 / (scen.y_max - scen.y_min)
            vel_slope = 2.0/(scen.vel_max-scen.vel_min)
            measure_slope = 1.0/np.pi
            distance_slope = 2.0/max_distance
            #normalization (map each value to the bound (-1,1)
            current_state[0] = -1+x_slope*(current_state[0]-scen.x_min)
            current_state[1] = -1 + y_slope * (current_state[1] - scen.y_min)
            current_state[2] = -1 + vel_slope * (current_state[2] - scen.vel_min)
            current_state[3] = -1 + vel_slope * (current_state[3] - scen.vel_min)
            current_state[4] = -1 + x_slope * (current_state[4] - scen.x_min)
            current_state[5] = -1 + y_slope * (current_state[5] - scen.y_min)
            current_state[6] = -1 + measure_slope*current_state[6]
            current_state[7] = -1 + distance_slope*current_state[7]

            #apply learnt RBF kernel and generate features
            current_state = featurizer.transform(np.array(current_state).reshape(1,len(current_state))) #apply this to RBF kernel
            current_state = list(current_state[0])
            for x in current_state:
                if x>1 or x<-1:
                    episode_condition = False
            #Update the location of sensor based on the weight and current state
            s.update_location(weight, sigma, np.array(current_state))
            #Store some important metrics, such as errors
            estimate = tracker_object.x_k_k
            episode_state.append(current_state)
            truth = t.current_location
            x_est.append(estimate[0])
            y_est.append(estimate[1])
            x_vel_est.append(estimate[2])
            y_vel_est.append(estimate[3])
            x_truth.append(truth[0])
            y_truth.append(truth[1])
            x_vel_truth.append(t.current_velocity[0])
            y_vel_truth.append(t.current_velocity[1])
            vel_error.append(np.linalg.norm(estimate[2:4]-np.array([t.current_velocity[0],t.current_velocity[1]]).reshape(2,1)))
            pos_error.append(np.linalg.norm(estimate[0:2]-np.array(truth).reshape(2,1)))
            innovation.append(normalized_innovation[0])

            unormalized_uncertainty = np.sum(tracker_object.p_k_k.diagonal())
            #reward: see if the uncertainy has decayed or it has gone below a certain value
            uncertainty.append((1.0/MAX_UNCERTAINTY)*unormalized_uncertainty)
            if len(uncertainty)<window_size+window_lag:
                reward.append(0)
            else:
                current_avg = np.mean(uncertainty[-window_size:])
                prev_avg = np.mean(uncertainty[-(window_size+window_lag):-window_lag])
                if current_avg<prev_avg or uncertainty[-1]<.1:
                #if current_avg < prev_avg:
                    reward.append(1)
                else:
                    reward.append(0)

            #Build discount vector
            discount_vector = gamma*np.array(discount_vector)
            discounted_return+= (1.0*reward[-1])*discount_vector
            new_return = 1.0*reward[-1]
            list_discounted_return = list(discounted_return)
            list_discounted_return.append(new_return)
            discounted_return = np.array(list_discounted_return)
            list_discount_vector = list(discount_vector)
            list_discount_vector.append(1)
            discount_vector = np.array(list_discount_vector)
            iteration.append(n)
            if n>episode_length: break
            n+=1
        #if episode_counter%10==0 and episode_counter>0: print(weight_saver[-1])
        prev_weight = np.array(weight)
        condition = True

        #Update weights only if the episode is valid
        if episode_condition:
            normalized_discounted_return = discounted_return
            episode_actions = s.sensor_actions #all sensor actions
            #init_weight = np.array(weight)
            rate = gen_learning_rate(episode_counter,learning_rate,1E-8,10000) #generate a new learning rate
            total_adjustment = np.zeros(np.shape(weight))
            #loop over all stored state/actions (trajectories)
            for e in range(0,len(episode_actions)):
                #calculate gradiant
                state = np.array(episode_state[e]).reshape(num_states,1)
                gradiant = ((episode_actions[e].reshape(2,1)-weight.dot(state)).dot(state.transpose()))/sigma**2#This is the gradiant
                if np.isnan(gradiant[0][0]) or np.linalg.norm(weight[0,:])>1E3:
                    condition = False
                    break
                adjustment_term = gradiant*normalized_discounted_return[e]#an unbiased sample of return (We replacce Q(s,a) by an estimate)
                weight+= rate*adjustment_term

            if not condition:
                weight = prev_weight
                continue

            return_saver.append(sum(reward))
            #print(discounted_return[500])
            if episode_counter%100==0 and episode_counter>0:
                print(episode_counter,np.mean(return_saver),sigma)
                avg_reward.append(np.mean(return_saver))
                var_reward.append(np.var(return_saver))
                return_saver = []
            episode_counter+=1
            weight_saver1.append(weight[0][0])
            weight_saver2.append(weight[0][1])
        else:
            #print("garbage trajectory: no-update")
            pass
        num_episodes.append(n)
        #print(weight)

    for r in avg_reward: writer_avg_reward.write(str(r)+"\n")
    for w in weight_saver1: writer_weight_update.write(str(w)+"\n")
    w1 = list(weight[0])
    w2 = list(weight[1])
    ww1 = []
    ww2 = []
    [ww1.append(str(x)) for  x in w1]
    [ww2.append(str(x)) for x in w2]
    writer_final_weight.write("\t".join(ww1)+"\n")
    writer_final_weight.write("\t".join(ww2))
    for v in var_reward: writer_var_reward.write(str(v)+"\n")
    writer_var_reward.close()
    writer_final_weight.close()
    writer_weight_update.close()
    writer_avg_reward.close()
    sys.exit(1)
    plt.subplot(4, 1, 1)
    plt1, = plt.plot(iteration, vel_error, "bs-", linewidth=3)
    plt.ylabel("Velocity Estimate Error", size=15)
    plt.grid(True)
    plt.subplot(4, 1, 2)
    plt2, = plt.plot(iteration, pos_error, "rd-", linewidth=3)
    plt.ylabel("Position Estimate Error", size=15)
    plt.grid(True)
    plt.subplot(4, 1, 3)
    plt3, = plt.plot(iteration, innovation, "mo-", linewidth=3)
    plt.xlabel("iteration", size=15)
    plt.ylabel("Predicted Innovation", size=15)
    plt.grid(True)

    plt.subplot(4, 1, 4)
    plt3, = plt.plot(iteration, reward, "ko-", linewidth=3)
    plt.xlabel("iteration", size=15)
    plt.ylabel("Reward function", size=15)
    plt.grid(True)
    plt.show()


    x_t = []
    y_t = []
    x_s = []
    y_s = []

    [x_t.append(x[0]) for x in t.historical_location]
    [y_t.append(x[1]) for x in t.historical_location]

    [x_s.append(x[0]) for x in s.historical_location]
    [y_s.append(x[1]) for x in s.historical_location]

    plt1, = plt.plot(x_t, y_t, "bs-", linewidth=3)
    plt2, = plt.plot(x_s, y_s, "ro--", linewidth=3)
    plt.xlabel("X", size=15)
    plt.ylabel("Y", size=15)
    plt.grid(True)
    plt.legend([plt1, plt2], ["Target", "Sensor"])
    plt.show()
    sys.exit(1)

    plt.subplot(3, 1, 1)
    plt1, = plt.plot(iteration, vel_error, "bs-", linewidth=3)
    plt.ylabel("Velocity Estimate Error", size=15)
    plt.grid(True)
    plt.subplot(3, 1, 2)
    plt2, = plt.plot(iteration, pos_error, "rd-", linewidth=3)
    plt.ylabel("Position Estimate Error", size=15)
    plt.grid(True)
    plt.subplot(3, 1, 3)
    plt3, = plt.plot(iteration, innovation, "mo-", linewidth=3)
    plt.xlabel("iteration", size=15)
    plt.ylabel("Predicted Innovation", size=15)
    plt.grid(True)
    plt.show()
    sys.exit(1)


    sys.exit(1)

        #tracker_object.update_states(s.current_location,m[-1])

        # plot both trajectories



    plt1, = plt.plot(x_truth,y_truth,"bs-",linewidth=3)
    plt2, = plt.plot(x_est, y_est, "rd-", linewidth=3)
    plt.xlabel("X", size=15)
    plt.ylabel("Y", size=15)
    plt.grid(True)
    plt.legend([plt1, plt2], ["Ground Truth", "Estimate"])
    plt.show()
    sys.exit(1)

    x_t = []
    y_t = []
    x_s = []
    y_s = []

    [x_t.append(x[0]) for x in t.historical_location]
    [y_t.append(x[1]) for x in t.historical_location]

    [x_s.append(x[0]) for x in s.historical_location]
    [y_s.append(x[1]) for x in s.historical_location]

    plt1, = plt.plot(x_t, y_t, "bs-", linewidth=3)
    plt2, = plt.plot(x_s, y_s, "ro--", linewidth=3)
    plt.xlabel("X", size=15)
    plt.ylabel("Y", size=15)
    plt.grid(True)
    plt.legend([plt1, plt2], ["Target", "Sensor"])
    plt.show()
    sys.exit(1)


