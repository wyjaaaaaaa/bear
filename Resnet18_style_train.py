import  tensorflow       as tf
from    tensorflow.keras import layers, optimizers, datasets, Sequential
import scipy.io as scio
import  matplotlib.pyplot as plt
import  numpy as np
import random
import  time
import h5py
import  os
from    resnet import resnet18
from PIL import Image


os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.random.set_seed(2345)

np.random.seed(2345)



def preprocess(x, y):
    # [-1~1]
    x = tf.cast(x, dtype=tf.float64)
    y = tf.cast(y, dtype=tf.int32)
    return x,y

###Test训练数据
test= h5py.File(r"E:\轴承数据\Resnet_data\Stylegan_Test_data.mat",'r')
x_test = test['x']
x_test = tf.transpose(x_test,perm=[3,1,2,0])
y_test0 = np.ones((320,1))*0
y_test1 = np.ones((800,1))*1
y_test2 = np.ones((800,1))*2
y_test = np.append(y_test0,y_test1,axis=0)
y_test = np.append(y_test,y_test2,axis=0)

###原始的正常训练数据
Normal_train= h5py.File(r"E:\轴承数据\Resnet_data\Stylegan_Normal_data.mat",'r')
x_Normal_train = Normal_train['x']
x_Normal_train = tf.transpose(x_Normal_train,perm=[3,1,2,0])
y_Normal_train = np.zeros((1500,1))  #6000个数据，因为大于7722组数据就超出内存

# ###生成的OR训练数据
OR1_train= h5py.File(r"E:\轴承数据\Resnet_data\Stylegan_Generate_OR_data.mat",'r')
x_OR_train1 = OR1_train['x']
x_OR_train1 = tf.transpose(x_OR_train1,perm=[3,1,2,0])
y_OR_train1 = np.ones((1500,1))  #6000个数据，因为大于7722组数据就超出内存
    ##原始的OR训练数据
OR2_train= h5py.File(r"E:\轴承数据\Resnet_data\Stylegan_Real_OR_data.mat",'r')
x_OR_train2 = OR2_train['x']
x_OR_train2 = tf.transpose(x_OR_train2,perm=[3,1,2,0])
y_OR_train2 = np.ones((1500,1))  #6000个数据，因为大于7722组数据就超出内存
    ##合并OR训练数据
x_OR_train=np.append(x_OR_train1,x_OR_train2,axis=0)
y_OR_train=np.append(y_OR_train1,y_OR_train2,axis=0)

###原始的IR训练数据
IR_train= h5py.File(r"E:\轴承数据\Resnet_data\Stylegan_IR_data.mat",'r')
x_IR_train = IR_train['x']
x_IR_train = tf.transpose(x_IR_train,perm=[3,1,2,0])
y_IR_train = np.ones((1500,1))*2  #6000个数据，因为大于7722组数据就超出内存

x_train = np.append(x_Normal_train,x_OR_train,axis=0)
x_train = np.append(x_train,x_IR_train,axis=0)

y_train = np.append(y_Normal_train,y_OR_train,axis=0)
y_train = np.append(y_train,y_IR_train,axis=0)



###进行切片打乱，防止验证集上的数据全是1,2种故障，没有3,4,5的故障
# c = list(zip(x_train, y_train))
# random.shuffle(c)
# x_train, y_train = zip(*c)

#对原始数据进行训练集与验证集的划分

y_train = tf.squeeze(y_train, axis=1)
y_test = tf.squeeze(y_test, axis=1)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

###数据和标签打包切片
train_db = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_db = train_db.shuffle(10000).map(preprocess).batch(1)
test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_db = test_db.map(preprocess).batch(1)

def main(loss_sum=None, acc_sum=None):

    # [6000, 256, 256, 3] => [6000, 1, 1, 512]
    model = resnet18()      #调用resnet()模型
    model.build(input_shape=(None, 256, 256,3))
    model.summary()
    optimizer = optimizers.Adam(lr=1e-5)
    accmax = 0
    accnum = 0
    pred_max = 0
    acc_sum = []  # 正确率汇总
    loss_sum = [] # 错误率汇总

    for epoch in range(500):
        time_start = time.time()
        #训练集
        for step, (x,y) in enumerate(train_db):

            with tf.GradientTape() as tape:
                # [b, 32, 32, 3] => [b, 100]
                logits = model(x)
                # [b] => [b, 100]
                y_onehot = tf.one_hot(y, depth=3)
                # compute loss
                loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        #验证集
        total_num = 0
        total_correct = 0
        i=0
        for x,y in test_db:
            i+=1
            logits = model(x)
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)
            if i==1:
                pred_max = prob
            else:
                pred_max =  np.append(pred_max,prob,axis=0)

            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)

            total_num += x.shape[0]
            total_correct += int(correct)

        acc = total_correct / total_num

        acc_sum.append(1)
        loss_sum.append(1)
        acc_sum[epoch]  = acc          #正确率汇总
        loss_sum[epoch] = loss         #错误率汇总

        time_end = time.time()

        print('------------------------------------------------------------------------------')
        print('第', epoch + 1, '次迭代损失：', float(loss))
        print('第', epoch + 1, '验证集准确率:', acc)
        print('第',epoch+1,'次迭代用时：',time_end - time_start)
        loss_out=float(loss)
        f.write('----------------------------------------------------------------------------\n'
                '第'+ str(epoch + 1)+'次迭代损失：'+ str(loss_out)+'\n'
                '第'+ str(epoch + 1)+'次验证集准确率:'+ str(acc)+'\n'
                '第'+ str(epoch + 1)+ '次迭代用时：'+ str(time_end - time_start)+'\n'
                )


        if acc>accmax:
            accmax=acc
            accnum=0

            scio.savemat('./损失与正确率数据/' + 'pred-y-gan'
                         + '.mat', {'pred': pred_max, 'y': y_test})
        else:

            accnum+=1

        if accnum>=20:
            f.write('----------------------------------------------------------------------------\n'
                     + '网络最大正确率：' + str(accmax) + '\n')
            #程序终止之前绘图保存
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()  # 产生一个ax1的镜面坐标
            lns1=ax1.plot(np.arange(epoch+1), loss_sum,"g-.")
            lns2=ax2.plot(np.arange(epoch+1), acc_sum,"b-")

            scio.savemat('./损失与正确率数据/'+'Resnet18_stylegan1'
                         +'.mat', {'Loss': loss_sum, 'Acc': acc_sum})

            ax1.set_xlabel("iteration")
            ax1.set_title('Resnet18')

            ax1.set_ylabel("training loss", color ="g")
            ax2.set_ylabel("Validation accuracy", color ="b")
            labels = ["Loss", "Accuracy"]
            # 合并图例
            lns = lns1 + lns2
            plt.legend(lns, labels, loc=7)

            #在图形上标注正确率最大点
            max_indx = np.argmax(acc_sum)
            show_max = '[' + str(max_indx) + ' ' + str(acc_sum[max_indx]) + ']'
            plt.annotate(show_max, xytext=(max_indx, acc_sum[max_indx]), xy=(max_indx, acc_sum[max_indx]))



            plt.savefig('./损失与正确率结果图/'+ 'Resnet18_style_train1'+'.jpg',dpi=500)
            # plt.show()

            break



if __name__ == '__main__':

        print('==========================', 'Resnet18', '==========================')
        timestart=time.time()
        f = open('./损失与正确率数据/' + 'Resnet18_style_train1' + '.txt', 'w')
        main()
        timeend=time.time()

        print('网络训练用时：', timeend - timestart)

        f.write('----------------------------------------------------------------------------\n'
                +'Resnet18_train'+'网络训练用时：'+ str(timeend - timestart)+'\n'
                )
        f.close()