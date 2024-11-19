import axios from 'axios';

const request = axios.create({
	baseURL: 'http://localhost:5010/api/', 
    timeout: 5000000
});
// const request = axios.create({
//     baseURL: 'https://1a55188ef199.ngrok.app.ngrok.io/api/',
//     timeout: 500000
//   });
  

// request 拦截器
// 可以自请求发送前对请求做一些处理
// 比如统一加token，对请求参数统一加密
request.interceptors.request.use(config => {
    config.headers['Content-Type'] = 'application/json;charset=utf-8';

 // config.headers['token'] = user.token;  // 设置请求头
    return config;
}, error => {
    return Promise.reject(error);
});

// response 拦截器
// 可以在接口响应后统一处理结果
request.interceptors.response.use(
    response => {
        let res = response.data;
        // 如果是返回的文件
        if (response.config.responseType === 'blob') {
            return res;
        }
        // 兼容服务端返回的字符串数据
        if (typeof res === 'string') {
            res = res ? JSON.parse(res) : res;
        }
        return res;
    },
    error => {
        console.log('err' + error); // for debug
        return Promise.reject(error);
    }
);


export default class requesthelp{
    static axiosGet(url,params = {},timeout,callback){
        return request({ method: 'get', url, params, timeout, callback });
    }
    static axiosPost(url, data = {},timeout,callback){
        return request({ method: 'post', url, data, timeout, callback });
    }
}


