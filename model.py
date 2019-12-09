import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed, may indicate binary incompatibility")
import tensorflow as tf
from CommonFunc import *
from ops import *
from Utils import *
from input_func import *
import Data.config as cfg
class YHMODEL:
    def __init__(self):
        self.NUM_CLASSES=2
        self.IMAGE_W=320
        self.IMAGE_H=416
        self.IMAGE_C=6 #
        self.BATCH_SIZE=2
        self.CHPT_DIR='logs'
        self.INITIAL_LEARNING_RATE=0.001
        self.EPOCH=5000
        self.func=CommonFunc()
        self.fsd_logit=None
        self.data_dir="/home/younghwajung/project/data/Total_data_set/"

        self.train_dir=self.data_dir+"training/"
        self.test_dir=self.data_dir+"test/"

        self.sess=None
        self.top_view_image=None
        self.bd_label_image=None


        self.sum_fsd_image=None
        self.sum_precision=None
        self.sum_recall=None
        self.summary_writer=None

        self.input_node=None
        self.curb_label_node=None
        self.train_flag=None
        self.summary_image_node=None
        self.precision_node=None
        self.recall_node=None

        self.val_interval=50
        self.visual_summary=50

        self.curb_infer=None

        self.detection_thres=0.5


    def curb_detection_based_on_line(self):#line based data
        gl_steps=0
        with tf.Graph().as_default():
            #place holder#
            self.train_flag = tf.placeholder(dtype=tf.bool, name='flag_placeholder')
            self.input_node = tf.placeholder(name='input', dtype=tf.float32,shape=[self.BATCH_SIZE, self.IMAGE_H, self.IMAGE_W, self.IMAGE_C])
            self.curb_label_node = tf.placeholder(name='label',dtype=tf.int32, shape=[self.BATCH_SIZE, self.IMAGE_H, self.IMAGE_W,1])
            self.summary_image_node=tf.placeholder(name="summary_image",dtype=tf.uint8,shape=[1,self.IMAGE_H, self.IMAGE_W, 3])

            self.test_image_node=tf.placeholder(name="test_image",dtype=tf.uint8,shape=[1,self.IMAGE_H, self.IMAGE_W, 3])
            self.precision_node = tf.placeholder(name='prec_node', dtype=tf.float32)
            self.recall_node = tf.placeholder(name='recall_node', dtype=tf.float32)
            self.test_precision_node = tf.placeholder(name='test_prec_node', dtype=tf.float32)
            self.test_recall_node = tf.placeholder(name='test_recall_node', dtype=tf.float32)
            #place holder#

            #train op#
            total_loss, self.curb_infer = self.loss_bd(self.input_node, self.curb_label_node, self.train_flag)
            #train op#

            #for bacth normalization
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op,cur_learing_rate = self.train(total_loss, global_step=gl_steps)
            #for bacth normalization


            #Summary#
            probabilities = tf.nn.softmax(self.curb_infer)
            with tf.name_scope("overall_statistic"):
                sum_to_loss=tf.summary.scalar("total_loss", total_loss)
                self.sum_learning_rate=tf.summary.scalar("learning_rate",cur_learing_rate)
            with tf.name_scope("accuracy"):
                self.sum_precision = tf.summary.scalar("train_precision", self.precision_node)
                self.sum_recall = tf.summary.scalar("train_recall", self.recall_node)
                self.test_precision = tf.summary.scalar("test_precision", self.test_precision_node)
                self.test_recall = tf.summary.scalar("test_recall", self.test_recall_node)
            with tf.name_scope("image_visual"):
                self.sum_fsd_image=tf.summary.image("image",self.summary_image_node,3)
                self.sum_test_image=tf.summary.image("test_image",self.test_image_node,3)
            self.summary_writer = tf.summary.FileWriter('logs', graph=tf.get_default_graph())
            # Summary#


            print("data loading")
            data_set, fsd_label_set, frame_number_list=self.func.bd_data_gen_6(self.train_dir)#
            number_set=[i for i in range(data_set.shape[0])]
            print(len(number_set))
            print("loading end")

            saver = tf.train.Saver()
            with tf.Session() as sess:
                self.sess=sess
                if tf.train.get_checkpoint_state(self.CHPT_DIR):
                    print("Reading model parameters from %s" % self.CHPT_DIR)
                    saver.restore(sess, tf.train.latest_checkpoint(self.CHPT_DIR))
                else:
                    print("start training")
                    sess.run(tf.global_variables_initializer())

                for epoch in range(self.EPOCH):
                    print(epoch)
                    random.shuffle(number_set)
                    for step in range(1, fsd_label_set.shape[0]//self.BATCH_SIZE+1):
                        lidar_data, fsd_label, visual_frame = self.func.shuffle_batch_data_bd(step, number_set, data_set,fsd_label_set, frame_number_list,batch_size=self.BATCH_SIZE)
                        feed_dict = {self.input_node: lidar_data, self.curb_label_node: fsd_label, self.train_flag: True}

                        _, tmp_total_loss, tmp_learning_rate= sess.run([train_op, sum_to_loss,self.sum_learning_rate], feed_dict=feed_dict)

                        gl_steps += 1

                    if epoch %1==0:
                        self.summary_writer.add_summary(tmp_total_loss, epoch)#total loss
                    if epoch %self.visual_summary==0:
                        self.top_view_image=cv2.imread(self.train_dir + "/top_view_raw/"+str(visual_frame)+'.png')#top-view image
                        self.bd_label_image=cv2.imread(self.train_dir+"/boundary_image/"+str(visual_frame)+".png",cv2.IMREAD_GRAYSCALE)#to calculate the precision and recall.
                        self.summary_writer.add_summary(tmp_learning_rate, epoch)
                        feed_input_list=[lidar_data,fsd_label,epoch]
                        self.draw_sum_line(probabilities,feed_input_list)
                        self.bd_point_based_PR(probabilities,feed_input_list)
                    if epoch%self.val_interval==0:
                        print("evaluating validation set...")
                        self.validating(probabilities,epoch)
                        saver.save(sess, os.path.join(self.CHPT_DIR, 'model_' + str(epoch)))
                        print('saving epoch %d, graph size : %d' % (epoch, sess.graph_def.ByteSize()))



    def train(self,total_loss, global_step):
        learning_rate = tf.train.exponential_decay(self.INITIAL_LEARNING_RATE, global_step,2000, 0.8, staircase=True)
        opt = tf.train.AdamOptimizer(learning_rate)
        train_operation = opt.minimize(loss=total_loss)
        return train_operation,learning_rate


    def loss_bd(self,images,bd_labels,train_flag):
        infer_fsd= self.inference_bd_2(images,Training=train_flag)
        logits = tf.reshape(infer_fsd, [-1, self.NUM_CLASSES])
        labels = tf.reshape(bd_labels, [-1])
        cross_entrophy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,name='cross_entrophy')
        cross_entrophy_sum = tf.reduce_sum(cross_entrophy,name='mean_cross_entrophy')
        total_loss=cross_entrophy_sum
        return total_loss,infer_fsd


    def draw_sum_bd_point(self, probabilities, feed_input_list):
        summary_feed_dict = {self.input_node: feed_input_list[0], self.curb_label_node: feed_input_list[1], self.train_flag: False}
        prob = self.sess.run(probabilities, feed_dict=summary_feed_dict)
        pro_image = np.squeeze(prob[0])
        raw_image = self.top_view_image.copy()
        for h in range(pro_image.shape[0]):
            for w in range(pro_image.shape[1]):
                if (pro_image[h, w, 1] > self.detection_thres):
                    cv2.circle(raw_image,(w,h),3,(255,0,0),-1)

        result_image = np.expand_dims(raw_image, axis=0)
        temp_feed = {self.summary_image_node: result_image}
        summary_temp = self.sess.run(self.sum_fsd_image, feed_dict=temp_feed)
        self.summary_writer.add_summary(summary_temp, feed_input_list[2])


    def draw_sum_line(self, probabilities, feed_input_list):
        summary_feed_dict = {self.input_node: feed_input_list[0], self.curb_label_node: feed_input_list[1], self.train_flag: False}
        prob = self.sess.run(probabilities, feed_dict=summary_feed_dict)
        pro_image = np.squeeze(prob[0])
        raw_image = self.top_view_image.copy()
        for h in range(pro_image.shape[0]):
            for w in range(pro_image.shape[1]):
                if (pro_image[h, w, 1] > self.detection_thres):
                    #cv2.circle(raw_image,(w,h),3,(255,0,0),-1)
                    raw_image[h,w,0]=255
                    raw_image[h,w, 1] = 0
                    raw_image[h, w, 2] = 0

        result_image = np.expand_dims(raw_image, axis=0)
        temp_feed = {self.summary_image_node: result_image}
        summary_temp = self.sess.run(self.sum_fsd_image, feed_dict=temp_feed)
        self.summary_writer.add_summary(summary_temp, feed_input_list[2])


    def validating(self,probabilties,gl_steps):
        vali_set, vali_label_set, frame_number_list = self.func.test_data_gen_2(self.test_dir,self.train_dir)#changing visual cue.
        number_set = [i for i in range(vali_set.shape[0])]
        random.shuffle(number_set)

        total_pre,total_recall=0,0
        for step in range(1, vali_label_set.shape[0] 
            input_data, labels,first_frame_number=self.gen_test_data_bd(step,number_set,vali_set,vali_label_set,frame_number_list,batch_size=2)

            vali_feed_dict={self.input_node:input_data, self.curb_label_node:labels,self.train_flag:False}


            prob = self.sess.run(probabilties, feed_dict=vali_feed_dict)

            precision, recall=self.vali_bd_point_based_PR(prob,labels,gl_steps)
            total_pre+=precision
            total_recall+=recall

            if(step==1):
                pro_image = np.squeeze(prob[0])  #
                raw_image = cv2.imread(self.train_dir + "/top_view_raw/" + str(first_frame_number) + '.png')
                for h in range(pro_image.shape[0]):
                    for w in range(pro_image.shape[1]):
                        if (pro_image[h, w, 1] > self.detection_thres):
                            #cv2.circle(raw_image, (w, h), 3, (255, 0, 0), -1)
                            raw_image[h, w, 0] = 255
                            raw_image[h, w, 1] = 0
                            raw_image[h, w, 2] = 0

                result_image = np.expand_dims(raw_image, axis=0)
                temp_feed = {self.test_image_node: result_image}
                summary_test_image = self.sess.run(self.sum_test_image, feed_dict=temp_feed)
                self.summary_writer.add_summary(summary_test_image, gl_steps)

        avg_pre=total_pre/vali_label_set.shape[0]#각 frame당 precision을 더해서
        avg_recall=total_recall/vali_label_set.shape[0]#각 frame당 recall을 더해서

        feed_for_acc={self.test_precision_node:avg_pre, self.test_recall_node:avg_recall}
        tmp_pre,tmp_recall = self.sess.run([self.test_precision,self.test_recall], feed_dict=feed_for_acc)
        self.summary_writer.add_summary(tmp_pre, gl_steps)
        self.summary_writer.add_summary(tmp_recall, gl_steps)


    def gen_test_data_bd(self, step, number_set, train_data, fsd_label, frame_num_list,batch_size=5):
        batch_number = number_set[(step - 1) * batch_size:step * batch_size]
        first_frame_number=frame_num_list[batch_number[0]]
        batch_data, batch_fsd= [], []
        for idx in batch_number:
            batch_data.append(train_data[idx])
            batch_fsd.append(fsd_label[idx])

        loaded_data = np.concatenate(batch_data, axis=0).reshape(-1, cfg.IMAGE_H, cfg.IMAGE_W, 6)
        loaded_fsd = np.concatenate(batch_fsd, axis=0).reshape(-1, cfg.IMAGE_H, cfg.IMAGE_W, 1)

        return loaded_data, loaded_fsd, first_frame_number



    def inference_bd_2(self, image_tensor, Training=True, Reuse=False):#reduce model size
        with tf.variable_scope("resnet"):
            #init two convlayer
            layer1 = conv2d_res(image_tensor,n_in=6,n_out=32,k=3,s=1,p='SAME',bias=False,scope='conv_init_1')#false에서 Ttue로 바R무
            layer2=batch_norm_res(layer1,n_out=32,phase_train=Training,scope='bn_init')
            layer3=tf.nn.relu(layer2,name='relu_init')

            #residual+pyramid
            layer7 = residual_group(layer3, 32, 64, residual_net_n=5, first_subsample=True, phase_train=Training,scope='group_1') #208x160
            deconv7= Deconv2DfromVoxelNet(64,128,3,(2,2),(0,0),layer7,training=Training,name="deconv7")#416x320
            layer8 = residual_group(layer7, 64, 128,residual_net_n=5, first_subsample=True, phase_train=Training, scope='group_2')#104x80
            deconv8= Deconv2DfromVoxelNet(128,128,3,(4,4),(0,0),layer8,training=Training,name="deconv8")#416x320
            layer9 = residual_group(layer8, 128, 256,residual_net_n=5, first_subsample=True, phase_train=Training, scope='group_3')#52x40
            deconv9= Deconv2DfromVoxelNet(256,128,3,(8,8),(0,0),layer9,training=Training,name="deconv9")#416x320
            layer10 = residual_group(layer9, 256, 512,residual_net_n=5, first_subsample=True, phase_train=Training, scope='group_4')#26x20
            deconv10= Deconv2DfromVoxelNet(512,128,3,(16,16),(0,0),layer10,training=Training,name="deconv10")#400x300


            #final
            merge_conv=tf.concat([deconv7,deconv8,deconv9,deconv10],-1)
            layer9 = elu(conv2d(merge_conv, output_dim=32, name='conv_test_1', Reuse=tf.AUTO_REUSE))
            layer10= elu(conv2d(layer9, output_dim=self.NUM_CLASSES, name='conv_test_2', Reuse=tf.AUTO_REUSE))
            #final


            freespace_final=tf.reshape(layer10, [-1, self.IMAGE_H, self.IMAGE_W, self.NUM_CLASSES], name="fsd_final_layer")


        return freespace_final



    def bd_point_based_PR(self,probabilities,feed_input_list):
        summary_feed_dict = {self.input_node: feed_input_list[0], self.curb_label_node: feed_input_list[1], self.train_flag: False}
        prob = self.sess.run(probabilities, feed_dict=summary_feed_dict)
        #prediction_image = np.squeeze(prob[0])#squeeze 굳이 안 해도 되네.
        prediction_image=prob[0]#여기서 2개의 이미지중 하나의 이미지만 선택함.

        for row in range(prediction_image.shape[0]):
            for col in range(prediction_image.shape[1]):
                if(prediction_image[row,col,1]>self.detection_thres):
                    prediction_image[row, col,1]= 1
                else:
                    prediction_image[row, col, 1] = 0

        TP, FP, FN, TN,gt_num = 0, 0, 0, 0,0
        gen_label_image=self.bd_label_image
        for row in range(gen_label_image.shape[0]):
            for col in range(gen_label_image.shape[1]):
                if (gen_label_image[row, col] == 1):
                    gt_num+=1
                    if (row != 0 and row != 415 and col != 0 and col != 319):
                        if (prediction_image[row, col,1] == 1 or prediction_image[row + 1, col,1] == 1 or prediction_image[
                            row, col + 1,1] == 1 or prediction_image[row - 1, col,1] == 1 or prediction_image[
                            row, col - 1,1] == 1 or prediction_image[row + 1, col - 1,1] == 1 or
                                prediction_image[row - 1, col + 1,1] == 1 or prediction_image[row + 1, col + 1,1] == 1 or
                                prediction_image[row - 1, col - 1,1] == 1):
                            TP += 1
                        else:
                            FN += 1

                else:
                    if (row != 0 and row != 415 and col != 0 and col != 319):
                        if (prediction_image[row, col,1] == 1):
                            FP += 1
                        else:
                            TN += 1

        #FP=FP-gt_num*8
        precision = TP / (TP + FP + 1e-10)
        recall = TP / (TP + FN)

        feed_for_acc={self.precision_node:precision, self.recall_node:recall}
        tmp_pre,tmp_recall = self.sess.run([self.sum_precision,self.sum_recall], feed_dict=feed_for_acc)
        self.summary_writer.add_summary(tmp_pre, feed_input_list[2])
        self.summary_writer.add_summary(tmp_recall, feed_input_list[2])

        return precision,recall


    def vali_bd_point_based_PR(self,prob,labels,gl_steps):
        total_precision, total_recall=0,0

        for image_idx in range(prob.shape[0]):
            prediction_image=None
            prediction_image=prob[image_idx]#여기서 2개의 이미지중 하나의 이미지만 선택함.

            for row in range(prediction_image.shape[0]):
                for col in range(prediction_image.shape[1]):
                    if(prediction_image[row,col,1]> self.detection_thres):
                        prediction_image[row, col,1] = 1
                    else:
                        prediction_image[row, col, 1] = 0

            TP, FP, FN, TN,gt_num = 0, 0, 0, 0,0
            gen_label_image=labels[image_idx]
            for row in range(gen_label_image.shape[0]):
                for col in range(gen_label_image.shape[1]):
                    if (gen_label_image[row, col] == 1):#curb 영역에 대해서
                        gt_num+=1
                        if (row != 0 and row != 415 and col != 0 and col != 319):
                            if (prediction_image[row, col,1] == 1 or prediction_image[row + 1, col,1] == 1 or prediction_image[
                                row, col + 1,1] == 1 or prediction_image[row - 1, col,1] == 1 or prediction_image[
                                row, col - 1,1] == 1 or prediction_image[row + 1, col - 1,1] == 1 or
                                    prediction_image[row - 1, col + 1,1] == 1 or prediction_image[row + 1, col + 1,1] == 1 or
                                    prediction_image[row - 1, col - 1,1] == 1):

                                TP += 1 
                            else:

                                FN += 1

                    else:
                        if (row != 0 and row != 415 and col != 0 and col != 319):
                            if (prediction_image[row, col,1] == 1):
                                FP += 1
                            else:
                                TN += 1

            #FP=FP-gt_num*8
            precision = TP / (TP + FP + 1e-10)
            recall = TP / (TP + FN)
            total_precision+=precision
            total_recall+=recall


        return total_precision,total_recall


    def vali_bd_point_based_PR_strict(self,prob,labels,gl_steps):
        total_precision, total_recall=0,0

        for image_idx in range(prob.shape[0]):
            prediction_image=None
            prediction_image=prob[image_idx]#여기서 2개의 이미지중 하나의 이미지만 선택함.

            for row in range(prediction_image.shape[0]):
                for col in range(prediction_image.shape[1]):
                    if(prediction_image[row,col,1]> self.detection_thres):
                        prediction_image[row, col,1] = 1
                    else:
                        prediction_image[row, col, 1] = 0

            TP, FP, FN, TN,gt_num = 0, 0, 0, 0,0
            gen_label_image=labels[image_idx]
            for row in range(gen_label_image.shape[0]):
                for col in range(gen_label_image.shape[1]):
                    if (gen_label_image[row, col] == 1):
                        gt_num+=1
                        if (row != 0 and row != 415 and col != 0 and col != 319):
                            if (prediction_image[row, col,1] == 1 or prediction_image[row + 1, col,1] == 1 or prediction_image[
                                row, col + 1,1] == 1 or prediction_image[row - 1, col,1] == 1 or prediction_image[
                                row, col - 1,1] == 1 or prediction_image[row + 1, col - 1,1] == 1 or
                                    prediction_image[row - 1, col + 1,1] == 1 or prediction_image[row + 1, col + 1,1] == 1 or
                                    prediction_image[row - 1, col - 1,1] == 1):

                                TP += 1
                            else:

                                FN += 1

                    else:
                        if (row != 0 and row != 415 and col != 0 and col != 319):
                            if (prediction_image[row, col,1] == 1):
                                FP += 1
                            else:
                                TN += 1

            #FP=FP-gt_num*8
            precision = TP / (TP + FP + 1e-10)
            recall = TP / (TP + FN)
            total_precision+=precision
            total_recall+=recall

        return total_precision,total_recall




