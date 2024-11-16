<template>
    <div class="principlerecommendationContainer viewbottom view">
        <div style="width: 100%; height: 10%; display: flex; align-items: flex-start; justify-content: flex-start">
            <div class="selectTitle" style="width: 95%;">Principle Recommendation</div>
            <el-button style="margin-right: 5px" type="info" @click="UploadItems()" :icon="DocumentAdd" round
                plain>Import </el-button>
        </div>
        <div class="principleView">
            <div v-for="(item, index) in kshotList" :key="index" class="card-wrapper" style="position: relative;">
                <div style="height:3px;position: relative;"></div>
                <el-card class="box-card" :body-style="{ padding: '10px' }" :class="{'high-background': item.type === 'high', 'low-background': item.type === 'low'}">
                    <div class="cardcontent" style="display: flex; align-items: center;">
                        <el-checkbox v-model="item.selected" style="padding-right: 15px"></el-checkbox>
                        <!-- <div style="width: 100%; font-size:15px; font-family:cursive;" v-if="!item.editing"> -->
                        <div style="width: 100%; font-size:18px; font-family: sans-serif" v-if="!item.editing">
                            {{ item.content }}
                        </div>
                        <el-input v-else type="textarea" v-model="item.content" class="inline-edit" autosize>
                        </el-input>
                        <el-button v-if="item.editing" link @click="editItem(item, index)">
                            <el-icon>
                                <Check />
                            </el-icon>
                        </el-button>
                        <el-button v-else link @click="editItem(item, index)">
                            <el-icon>
                                <Edit />
                            </el-icon>
                        </el-button>
                        <el-button link @click="deleteItem(index)">
                            <el-icon>
                                <Delete />
                            </el-icon>
                        </el-button>
                    </div>
                </el-card>

                <span v-if="item.ifnew" class="new-tag"></span>
            </div>
        </div>

        <div class="input-with-button">
            <el-input class="principleinput" type="textarea" placeholder="Enter your principle here..."
                v-model="principle">
            </el-input>
            <el-button @click="uploadNew(index)" type="primary" plain class="uploadbutton">
                <el-icon class="el-icon--right">
                    <Promotion />
                </el-icon>
            </el-button>

        </div>
    </div>
</template>

<script setup>
import {
    ElCard,
    ElInput,
    ElButton,
    ElCheckbox,
    ElIcon
} from 'element-plus';
import { DocumentAdd, Promotion, Edit, Check, Delete } from '@element-plus/icons-vue'
</script>

<script>
import requesthelp from "../service/request.js";
import '@fortawesome/fontawesome-free/css/all.css';


export default {
    props: ['selectKshotData'],
    data() {
        return {
            principleData: {},
            kshotList: [],
            principle: "",
            uploaddata: []
        };
    },
    mounted() {
        this.getKshotExample()
    },
    watch: {
        'kshotList': {
            handler: 'onKshotListChanged',
            deep: true
        },
        selectKshotData() {
            this.GenerateItems()
        }
    },
    methods: {
        switchNew(){
            for (let i = 0; i < this.kshotList.length; i++){
                this.kshotList[i].ifnew = false
            }
        }, 
        uploadNew() {
            this.switchNew()
            this.kshotList.unshift({
                content: this.principle,
                editing: false,
                selected: false,
                type: "default",
                ifnew: true
            })
        },
        async GenerateItems() {
            var results = await requesthelp.axiosPost("generate_principle_list_with_instances", {
                video_list: this.selectKshotData
            })

            //static
            // var results = {
            //     "high_level_principle_list": [
            //         "Evaluate the overall context when analyzing sentiment, recognizing that serious or neutral expressions can coexist with positive or negative sentiment, particularly when verbal content is explicitly positive or negative.",
            //         "Give appropriate weight to the intensity of certain words in verbal content, such as 'incredible,' as they can significantly influence the tone and sentiment of the message.",
            //         "Focus on the interplay between visual and verbal cues, ensuring that the context of the spoken content is given primary importance in determining the overall sentiment, rather than attributing sentiment to visual cues in isolation."
            //     ],
            //     "low_level_principle_list": [
            //         "When analyzing sentiment, it is important to consider the context in which visual and verbal cues are presented. A serious expression does not necessarily equate to neutrality, especially when accompanied by a smile and positive speech. The principle to improve future responses is to evaluate the overall context and to recognize that a serious demeanor can coexist with positive sentiment, particularly when the verbal content is explicitly positive. Additionally, the intensity of certain words (like 'incredible') should be given more weight in determining sentiment, as they can significantly influence the tone of the message.",
            //         "When analyzing sentiment, it is crucial to consider the context in which visual cues and spoken content are presented. A neutral expression can be interpreted differently depending on the accompanying verbal message. In cases where the spoken content is negative, a neutral expression can serve to underscore the negative sentiment rather than contradict it. Future responses should focus on the interplay between visual and verbal cues, ensuring that the context of the spoken content is given primary importance in determining the overall sentiment. Additionally, it is important to avoid attributing sentiment to visual cues without considering the context provided by the verbal content."
            //     ]
            // }
            console.log("generated results", results)

            this.switchNew()
            for (let i = 0; i < results["low_level_principle_list"].length; i++) {
                this.kshotList.unshift({
                    content: results["low_level_principle_list"][i],
                    editing: false,
                    selected: false,
                    type: "low",
                    ifnew: true
                })
            }
            for (let i = 0; i < results["high_level_principle_list"].length; i++) {
                this.kshotList.unshift({
                    content: results["high_level_principle_list"][i],
                    editing: false,
                    selected: false,
                    type: "high",
                    ifnew: true
                })
            }
        },
        UploadItems() {
            this.$emit('principledata', this.uploaddata);
        },
        onKshotListChanged() {
            var uploaddata = []
            this.kshotList.forEach(item => {
                if (item.selected) {
                    uploaddata.push(item.content)
                }
            });
            this.uploaddata = uploaddata
        },
        async getKshotExample() {
            var results = await requesthelp.axiosGet("load_principle_list")
            this.principleData = results["principle_list"];
            for (let k = 0; k < this.principleData.length; k++) {
                this.kshotList.push({
                    // title: key,
                    content: this.principleData[k],
                    editing: false,
                    selected: false,
                    type: "default",
                    ifnew: false
                })
            }
        },
        editItem(item, index) {
            if (item.editing) {
                console.log('Saving item at index:', index, 'with content:', item.content);
            }
            item.editing = !item.editing;
        },
        deleteItem(index) {
            // Your delete logic here
            console.log('Deleting item at index:', index);
            this.kshotList.splice(index, 1);
        },
    }
}

</script>

<style>
.principlerecommendationContainer {
    height: 38vh;
    /* overflow-x: auto;   */
    /* padding: 5px; */
}

.el-icon--right {
    margin-right: 5px;
}

.uploadbutton {
    margin-right: 5px;
    margin-left: 5px;
}

.cardcontent {
    display: flex;
    min-height: 50px;
}

.principleView {
    /* principle generation  */
    overflow-x: auto;
    /* margin-bottom: 5px; */
    padding: 5px;
    height: calc(85% - 30px)
        /* border: 1px solid #ddd; */
}

.input-with-button {
    display: flex;
    /* Use flexbox to lay out children inline */
    align-items: flex-start;
    /* Align items vertically */
    height: 23px;
    margin-top: 5px;
}

.principleinput {
    height: 24px;
    /* padding: 2px; */
    margin-left: 5px;
}


.principleinput .el-textarea__inner {
    height: 100%;
    width: 100%;
    padding: 5px;
}

.promptContainer {
    height: 50%;
}

.box-card {
    margin-bottom: 5px;
    margin-left:5px ;
    margin-right:5px ;
    border: 1px solid #dcdcdc !important; /* 设置边框颜色和宽度 */
}

/* .principleView .el-card {
    --el-card-border-color: rgb(0, 0, 0) !important;
} */

.principleView .el-card__header {
    font-weight: bold !important;
    padding: 0px !important;
    text-align: left;
    border-bottom: 0;
}

.principleView .el-card__body {
    text-align: left;
    padding: 2px 10px !important;
}

.card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.card-header-actions {
    margin-left: auto;
}

.inline-edit .el-textarea__inner {
    flex-grow: 1;
    /* 输入框占据可用空间 */
    margin-right: 10px;
    font-size: smaller;
    /* 在输入框和按钮之间添加一些空间 */
    /* 不要设置固定的高度 */
}

.high-background {
    background-color: #E6EFED !important;
    /* background-color: #CCE0DA !important; */
}

.low-background {
    background-color: #f5f5f5 !important;
}

.new-tag {
    position: absolute; /* 使用绝对定位 */
    z-index: 1; /* 确保它在卡片内容之上 */
    top: 0px; /* 距顶部的距离 */
    right: 0px; /* 距右侧的距离 */
    width: 10px; /* Size of the dot */
    height: 10px; /* Size of the dot */
    border-radius: 50%; /* Make it round */
    background-color: #F94D4B; /* Red dot, change the color as needed */
}
</style>