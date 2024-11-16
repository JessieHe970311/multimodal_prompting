import axios from 'axios'
const GET_REQUEST = 'get'
const POST_REQUEST = 'post'
const backendPort = 5011
// const dataServerUrl = `http://127.0.0.1:${backendPort}/api`
const dataServerUrl = `https://c5176a5fb298.ngrok.app/api/`
// const backendPort = 5010;
// const dataServerUrl = `http://3.23.79.160:${backendPort}/api`;


function request(url, params, type, callback) {
    let func
    if (type === GET_REQUEST) {
        func = axios.get
    } else if (type === POST_REQUEST) {
        func = axios.post
    }

    func(url, params).then((response) => {
            if (response.status === 200) {
                callback(response["data"])
            } else {
                console.error(response) /* eslint-disable-line */
            }
        })
        .catch((error) => {
            console.error(error) /* eslint-disable-line */
        })
}

export default {
    dataServerUrl,
}