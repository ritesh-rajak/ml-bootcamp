class base_class:
    def show_train_data(self,data):
        import numpy
        import pandas
        import matplotlib.pyplot
        n=data.shape[1]
        m=data.shape[0]
        m=int(m*0.8)
        y=data.iloc[0:m,(n-1)]
        data=data.iloc[0:m,:]
        data_of_x=data.to_numpy()
        data_of_x=numpy.delete(data_of_x,0,1)
        data_of_x=numpy.delete(data_of_x,(n-2),1)
        print(data_of_x)
        return data_of_x,y
    def show_test_data(self,data):
        import numpy
        import pandas
        import matplotlib.pyplot
        n=data.shape[1]
        m=data.shape[0]
        a=int(m*0.8)
        y=data.iloc[a:m,(n-1)]
        data=data.iloc[a:m,:]
        data_of_x=data.to_numpy()
        data_of_x=numpy.delete(data_of_x,0,1)
        data_of_x=numpy.delete(data_of_x,(n-2),1)
        print(data_of_x)
        return data_of_x,y
    def plot_y_versus_x(self,data1,data2,y1,y2,n):
        import numpy
        import pandas
        import matplotlib.pyplot
        for i in range(n):
            matplotlib.pyplot.grid(True)
            matplotlib.pyplot.xlabel(i+1)
            matplotlib.pyplot.ylabel('y')
            matplotlib.pyplot.scatter(data1[:,i],y1)
            matplotlib.pyplot.scatter(data2[:,i],y2)
            matplotlib.pyplot.show()
    def z_scoring(self,data,m,n):
        import numpy
        import pandas
        import matplotlib.pyplot
        avg=0
        standard_deviation=0
        for i in range(n):
            avg=((numpy.sum(data[:,i]))/m)
            standard_deviation=numpy.std(data[:,i])
            data[:,i]-=avg
            data[:,i]/=standard_deviation
        print(data)
        return data
    def calculate_predicted_label(self,w,b,data,m):
        import numpy
        import pandas
        import matplotlib.pyplot
        y_hat=numpy.matmul(w,data.transpose())+b
        y_hat=numpy.reshape(y_hat,m)
        return y_hat
    def gradient_dissent(self,w,b,y_hat,y,learn,data,m,n):
        import numpy
        import pandas
        import matplotlib.pyplot
        data_of_all_grad_w=0
        grad_b=(1/m)*(numpy.sum(y_hat-y))
        temp_w=numpy.zeros(n)
        for i in range(n):
            data_of_all_grad_w=(1/m)*(numpy.sum((y_hat-y)*data[:,i]))
            temp_w[i]=w[i]-(learn)*data_of_all_grad_w
        temp_b=b-((learn)*(grad_b))
        w=temp_w
        b=temp_b
        return w,b
    def cost_function_for_l_and_p(self,y_hat,y,m):
        import numpy
        import pandas
        import matplotlib.pyplot
        cost_function=(1/2)*(1/m)*(numpy.sum((y_hat-y)**2))
        return cost_function
    def iterations(self,no_of_iters,learn,y_hat,y,data,m,n):
        import numpy
        import pandas
        import matplotlib.pyplot as plt
        w_new=numpy.zeros(n)
        b_new=0
        cost_data=numpy.zeros(no_of_iters)
        no_of_data=numpy.zeros(no_of_iters)
        for iters in range(no_of_iters):
            w_new,b_new=self.gradient_dissent(w_new,b_new,y_hat,y,learn,data,m,n)
            y_hat=self.calculate_predicted_label(w_new,b_new,data,m)
            no_of_data[iters]=iters
            cost_data[iters]=self.cost_function_for_l_and_p(y_hat,y,m)
            if((iters+1)%10==0):
                print(iters+1)
                print("The cost function is:",cost_data[iters])
        plt.grid(True)
        plt.plot(no_of_data,cost_data)
        return w_new,b_new
    def feature_scaling_for_logistic(self,data):
        data=data/255
        return data
    def show_test_classification_data(self,data):
        import numpy
        import pandas
        import matplotlib.pyplot
        y=data.iloc[24000:30000,1]
        y=y.to_numpy()
        data=data.iloc[24000:30000,:]
        data_of_x=data.to_numpy()
        data_of_x=numpy.delete(data_of_x,0,1)
        data_of_x=numpy.delete(data_of_x,0,1)
        print(data_of_x)
        print(y)
        return data_of_x,y
    def r2(self,y_hat,y_actual):
        import numpy as np
        y_up=np.sum((y_hat-y_actual)**2)
        y_mean=np.mean(y_actual)
        y_actual=y_actual-y_mean
        y_down=np.sum((y_actual)**2)
        r_2=1-((y_up)*((y_down)**(-1)))
        return r_2
class linear_regression(base_class):
    def apply_linear_regression(self,data1,data2,iters,learn):
        import numpy
        import pandas
        import matplotlib.pyplot
        data_of_x,y=self.show_train_data(data1)
        m=data_of_x.shape[0]
        n=data_of_x.shape[1]
        data_of_x=self.z_scoring(data_of_x,m,n)
        w,b=self.iterations(iters,learn,numpy.zeros(m),y,data_of_x,m,n)
        print("the final values of w and b are",w,"and",b,"respectively")
        data_of_test_x,y_actual=self.show_test_data(data2)
        m=data_of_test_x.shape[0]
        n=data_of_test_x.shape[1]
        data_of_test_x=self.z_scoring(data_of_test_x,m,n)
        y_pred=self.calculate_predicted_label(w,b,data_of_test_x,m)
    #    self.plot_y_versus_x(data_of_test_x,data_of_x,y_pred,y,n)
        predicted_data=pandas.DataFrame(y_pred,columns=['predicted value'])
        print("the final values of predicted y are",predicted_data)
        predicted_data.to_csv('predicted_linear_data.csv')
        r_2=self.r2(y_pred,y_actual)
        print("the r2 score is",r_2)
class polynomial_regression(base_class):
    def get_degreed_terms(self,degree,data):
        import numpy as np
        data_of_degreed_X=np.zeros([data.shape[0],0])
        for i in range((degree+1)):
            for j in range((degree+1)):
                for k in range((degree+1)):
                    if (i+j+k)>degree or (i+j+k)==0:
                        continue
                    else:
                        data_of_degreed_X=np.hstack((data_of_degreed_X,np.reshape((data[:,0]**k)*(data[:,1]**j)*(data[:,2]**i),(data.shape[0],1))))              
                        temp=0
        return data_of_degreed_X
    def apply_polynomial_regression(self,data1,data2,iters,learn,degree):
        import numpy
        import pandas
        import matplotlib.pyplot
        data_of_x,y=self.show_train_data(data1)
        m=data_of_x.shape[0]
        n=data_of_x.shape[1]
        data_of_x=self.z_scoring(data_of_x,m,n)
        data_of_x=self.get_degreed_terms(degree,data_of_x)
        m=data_of_x.shape[0]
        n=data_of_x.shape[1]
        w,b=self.iterations(iters,learn,numpy.zeros(data_of_x.shape[0]),y,data_of_x,m,n)
        print("the final values of w and b are",w,"and",b,"respectively")
        data_of_test_x,y_actual=self.show_test_data(data2)
        m=data_of_test_x.shape[0]
        n=data_of_test_x.shape[1]
        data_of_test_x=self.z_scoring(data_of_test_x,m,n)
        data_of_test_x=self.get_degreed_terms(degree,data_of_test_x)
        m=data_of_test_x.shape[0]
        n=data_of_test_x.shape[1]
        y_pred=self.calculate_predicted_label(w,b,data_of_test_x,m)
      #  self.plot_y_versus_x(data_of_test_x,data_of_x,y_pred,y,n)           
        predicted_data=pandas.DataFrame(y_pred,columns=['predicted value'])
        print("the final values of predicted y are",predicted_data)
        predicted_data.to_csv('predicted_linear_data.csv')
        r_2=self.r2(y_pred,y_actual)
        print("the r2 score is",r_2)
class logistic_regression(base_class):
    def show_classification_data_for_logistic(self,data):
        import numpy
        import pandas
        import matplotlib.pyplot
        y=numpy.zeros((24000,10))
        for i in range(10):
            y[:,i]=data.iloc[0:24000,1].copy().to_numpy()
        data_new=data.iloc[0:24000]
        data_of_x=data_new.to_numpy()
        data_of_x=numpy.delete(data_of_x,0,1)
        data_of_x=numpy.delete(data_of_x,0,1)
        print(data_of_x)
        return data_of_x,y
    def finding_sigmoid(self,w,b,data,m):
        import numpy as np
        y_hat=((1+np.exp(-self.calculate_predicted_label(w,b,data,m)))**(-1))
        return y_hat
    def one_vs_all_data_classification(self,data):
        for i in range(10):
            for j in range(data.shape[0]):
                if(data[j,i]==i):
                    data[j,i]=1
                else:
                    data[j,i]=0
            print(data[:,i])
        return data
    def cost_function_for_logistic(self,y_hat,y,m):
        import numpy as np
        cost_function=-(1/m)*(np.sum((y*(np.log(y_hat)))+((1-y)*(np.log(1-y_hat)))))
        print("The cost function is:",cost_function)
        return y_hat
    def log_iters(self,no_of_iters,learn,y_hat,y,data,m,n):
        import numpy
        import pandas
        import matplotlib.pyplot
        w_new=numpy.zeros(n)
        b_new=0
        for iters in range(no_of_iters):
            w_new,b_new=self.gradient_dissent(w_new,b_new,y_hat,y,learn,data,m,n)
            y_hat=self.finding_sigmoid(w_new,b_new,data,m)
            if((iters+1)%50==0):
                print(iters+1)
                self.cost_function_for_logistic(y_hat,y,m)
        return w_new,b_new  
    def max_function(self,w,data,b,m):
        import numpy as np
        max_sigmoid=np.zeros(m)
        y_pred=np.zeros(m)
        for i in range(m):
            y_pred[i]=np.argmax((np.dot(w[0],data[i])+b[0],np.dot(w[1],data[i])+b[1],np.dot(w[2],data[i])+b[2],np.dot(w[3],data[i])+b[3],np.dot(w[4],data[i])+b[4],np.dot(w[5],data[i])+b[5],np.dot(w[6],data[i])+b[6],np.dot(w[7],data[i])+b[7],np.dot(w[8],data[i])+b[8],np.dot(w[9],data[i])+b[9]))
        return y_pred
    def apply_logistic_regression(self,data_train,data_test,iters,learn):
        import numpy
        import pandas
        import matplotlib.pyplot
        data_1,y=self.show_classification_data_for_logistic(data_train)
        m=data_1.shape[0]
        n=data_1.shape[1]
        data1=self.feature_scaling_for_logistic(data_1)
        indivi_data=self.one_vs_all_data_classification(y)
        print(indivi_data.shape)
        w=numpy.zeros((10,784))
        b=numpy.zeros(10)
        for i in range(10):
            w[i],b[i]=self.log_iters(iters,learn,numpy.zeros(m),indivi_data[:,i],data1,m,n)
        print("the final values of w and b are",w,"and",b,"respectively")
        data_2,y_actual=self.show_test_classification_data(data_test)
        data2=self.feature_scaling_for_logistic(data_2)
        m=data2.shape[0]
        n=data2.shape[1]
        y_pred=self.max_function(w,data2,b,m)  
      #  self.plot_y_versus_x(data_of_test_x,data_of_x,y_pred,y,n)           
        predicted_data=pandas.DataFrame(y_pred,columns=['predicted value'])
        print("the final values of predicted y are",predicted_data)
        predicted_data.to_csv('predicted_logistic_data.csv')
        accuracy=0
        for i in range(6000):
            if(y_actual[i]==y_pred[i]):
                accuracy+=1
        print("the accuracy is",accuracy/60,"%")
class KNN(base_class):
    def show_classification_data_for_knn(self,data):
        import numpy
        import pandas
        import matplotlib.pyplot
        y=data.iloc[0:24000,1]
        y=y.to_numpy()
        data=data.iloc[0:24000,:]
        data_of_x=data.to_numpy()
        data_of_x=numpy.delete(data_of_x,0,1)
        data_of_x=numpy.delete(data_of_x,0,1)
        print(data_of_x)
        return data_of_x,y
    def knn_op(self,data_train,data_test,m,n):
        import numpy as np
        data_test=np.expand_dims(data_test,axis=0)
        data_mat=np.repeat(data_test,m,axis=0)
        dist=np.sum(((data_mat-data_train)**2),axis=1)
        return dist
    def apply_KNN(self,k,data_train,data_test):
        import numpy as np
        data_of_x,y=self.show_classification_data_for_knn(data_train)
        testdata_of_x,y_actual=self.show_test_classification_data(data_test)
        m=data_of_x.shape[0]
        n=data_of_x.shape[1]
        m_test=testdata_of_x.shape[0]
        accuracy=0
        y_pred=np.zeros(m_test)
        final_array=np.zeros(k)
        for i in range(m_test):
            print(i)
            temp=self.knn_op(data_of_x,testdata_of_x[i],m,n)
            label_index=np.argsort(temp)
            label_index=label_index[0:k]
            for j in range(k):
                final_array[j]=y[label_index[j]]
            print(final_array)
            vals,counts=np.unique(final_array,return_counts=True)
            mode=np.argmax(counts)
            y_pred[i]=vals[mode]
            print(vals[mode])
            if(y_pred[i]==y_actual[i]):
                accuracy+=1
        print("the accuracy is",accuracy/60,"%")
class single_layer_neural_network(logistic_regression):
    def show_data(self,data):
        import numpy as np
        y=data.iloc[0:24000,1]
        y=y.to_numpy()
        y=np.expand_dims(y,axis=1)
        y=np.repeat(y,10,axis=1)
        data=data.iloc[0:24000,:]
        data_of_x=data.to_numpy()
        data_of_x=np.delete(data_of_x,0,1)
        data_of_x=np.delete(data_of_x,0,1)
        data_of_x=data_of_x/255
        print(data_of_x.shape)
        print(y.shape)
        return data_of_x,y
    def show_testdata(self,data):
        import numpy as np
        y=data.iloc[24000:30000,1]
        y=y.to_numpy()
        data=data.iloc[24000:30000,:]
        data_of_x=data.to_numpy()
        data_of_x=np.delete(data_of_x,0,1)
        data_of_x=np.delete(data_of_x,0,1)
        data_of_x=data_of_x/255
        print(data_of_x.shape)
        print(y.shape)
        return data_of_x,y
    def calculate_z(self,w,b,data):
        import numpy as np
        y_hat=np.matmul(w,data.T)+b
        y_hat=y_hat.T
        return y_hat
    def sigmoid(self,z):
        import numpy as np
        sig=(1+np.exp(-z))**(-1)
        return sig
    def sig_der(self,z):
        import numpy as np
        der=self.sigmoid(z)*(1-self.sigmoid(z))
        return der
    def softmax(self,z):
        import numpy as np
        num=np.exp(z)
        den=np.sum(num,axis=1)
        den=np.expand_dims(den,axis=1)
        soft=num/den
        return soft
    def relu(self,z):
        import numpy as np
        re=np.maximum(z,0)
        return re
    def relu_der(self,z):
        import numpy as np
        return np.array(z>0,dtype=np.float32)
    def tanh(self,z):
        import numpy as np
        return np.tanh(z)
    def tanh_der(self,z):
        import numpy as np
        return 1-((np.tanh(z))**2)
    def cost(self,data1,data2):
        import numpy as np
        m=data1.shape[0]
        cost_f=(1/m)*np.sum(-data1*np.log(data2))
        return cost_f
    def forward_prop(self,w1,w2,b1,b2,data):
        z1=self.calculate_z(w1,b1,data)
        a1=self.sigmoid(z1)
        z2=self.calculate_z(w2,b2,a1)
        a2=self.softmax(z2)
        return z1,a1,z2,a2
    def back_prop(self,w1,w2,b1,b2,z1,z2,a1,a2,indivi_data,data,alpha):
        import numpy as np
        m=data.shape[0]
        temp2=(a2-indivi_data)
        temp1=((np.matmul((a2-indivi_data),w2))*self.sig_der(z1))
        dj_dw2=(np.matmul(a1.T,temp2).T)
        dj_dw1=(np.matmul(data.T,temp1).T)
        dj_db2=np.sum(temp2,axis=0)
        dj_db1=np.sum(temp1,axis=0)
        for i in range(b2.shape[0]):
            b2[i]=b2[i]-((1/m)*alpha*dj_db2[i])
        for i in range(b1.shape[0]):
            b1[i]=b1[i]-((1/m)*alpha*dj_db1[i])
        w2=w2-(alpha*dj_dw2)
        w1=w1-(alpha*dj_dw1)
        return w1,b1,w2,b2
    def apply_single_layer_neural_network(self,data,neurons,iters,alpha):
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        cost_data=np.zeros(iters)
        no_of_iters=np.zeros(iters)
        data_x,y=self.show_data(data)
        indivi_data=self.one_vs_all_data_classification(y)
        np.random.seed(42)
        w1=np.random.randn(neurons,784)
        w2=np.random.randn(10,neurons)
        b1=np.random.randn(neurons,1)
        b2=np.random.randn(10,1)
        for i in range(iters):
            z1,a1,z2,a2=self.forward_prop(w1,w2,b1,b2,data_x)
            cost_data[i]=self.cost(indivi_data,a2)
            if((i+1)%10==0):
                print(i+1)
                no_of_iters[i]=i
                print(cost_data[i])
            w1,b1,w2,b2=self.back_prop(w1,w2,b1,b2,z1,z2,a1,a2,indivi_data,data_x,0.0003)
        plt.grid(True)
        plt.plot(no_of_iters,cost_data)
        testdata_x,y_actual=self.show_testdata(pd.read_csv('classification_train.csv'))
        z1,a1,z2,a2=self.forward_prop(w1,w2,b1,b2,testdata_x)
        y_pred=np.zeros(6000)
        accuracy=0
        for i in range(6000):
            y_pred[i]=np.argmax(a2[i])
            if(y_pred[i]==y_actual[i]):
                accuracy+=1
        print(accuracy/60)
class n_neural_network(single_layer_neural_network):
    def forward_prop_for_n(self,n,w,b,data):
        z={}
        a={}
        a[0]=data
        if(n%2!=0):
            for i in range(n):
                if(i%2==0):
                    z[i]=self.calculate_z(w[i],b[i],a[i])
                    a[i+1]=self.tanh(z[i])
                else:
                    z[i]=self.calculate_z(w[i],b[i],a[i])
                    a[i+1]=self.tanh(z[i])
        else:
            for i in range(n):
                if(i%2==0):
                    z[i]=self.calculate_z(w[i],b[i],a[i])
                    a[i+1]=self.tanh(z[i])
                else:
                    z[i]=self.calculate_z(w[i],b[i],a[i])
                    a[i+1]=self.tanh(z[i])
        z[n]=self.calculate_z(w[n],b[n],a[n])
        a[n+1]=self.softmax(z[n])
        return z,a
    def back_prop_for_n(self,n,w,b,z,a,indivi_data,alpha):
        import numpy as np
        dj_dw={}
        dj_db={}
        temp=a[n+1]-indivi_data
        m=24000
        for i in range(n+1):
            if(i%2==0):
                dj_dw[n-i]=np.matmul(temp.T,a[n-i])
                dj_db[n-i]=temp
                dj_db[n-i]=np.sum(dj_db[n-i],axis=0)
                if(i<n):
                    temp=np.matmul(temp,w[n-i])*self.tanh_der(z[n-1-i])
            else:
                dj_dw[n-i]=np.matmul(temp.T,a[n-i])
                dj_db[n-i]=temp
                dj_db[n-i]=np.sum(dj_db[n-i],axis=0)
                if(i<n):
                    temp=np.matmul(temp,w[n-i])*self.tanh_der(z[n-1-i])
        for i in range(n):
            w[i]=w[i]-alpha*(1/m)*dj_dw[i]
            for j in range(int(b[i].shape[0])):
                b[i][j]=b[i][j]-alpha*(1/m)*dj_db[i][j]
        return w,b
    def apply_n_layer_neural_network(self,data,n,iters,alpha):
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        w={}
        b={}
        np.random.seed(42)
        temp=np.zeros(n+2)
        temp[0]=784
        temp[n+1]=10
        for i in range(n+1):
            if(i<n):
                a="enter number of neurons in layer "+str(i+1)+":"
                temp[i+1]=int(input(a))
            w[i]=np.random.randn(int(temp[i+1]),int(temp[i]))
            b[i]=np.random.randn(int(temp[i+1]),1)
        cost_data=np.zeros(iters)
        no_of_iters=np.zeros(iters)
        data_x,y=self.show_data(data)
        indivi_data=self.one_vs_all_data_classification(y)
        # w1=np.random.randn(28,784)
        # w2=np.random.randn(10,28)
        # b1=np.random.randn(28,1)
        # b2=np.random.randn(10,1)
        for i in range(iters):
            z,a=self.forward_prop_for_n(n,w,b,data_x)
            no_of_iters[i]=i
            cost_data[i]=self.cost(indivi_data,a[n+1])
            if((i+1)%10==0):
                print(i+1)
                print(cost_data[i])
            w,b=self.back_prop_for_n(n,w,b,z,a,indivi_data,alpha)
        plt.grid(True)
        plt.plot(no_of_iters,cost_data)
        y_pred=np.zeros(6000)
        testdata_x,y_actual=self.show_testdata(data)
        z,a=self.forward_prop_for_n(n,w,b,testdata_x)
        accuracy=0
        for i in range(6000):
            y_pred[i]=np.argmax(a[n+1][i])
            if(y_pred[i]==y_actual[i]):
                accuracy+=1
        print(accuracy/60)