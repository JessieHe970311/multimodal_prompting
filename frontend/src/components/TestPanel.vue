<template>
    <div class="testpanelContainer view">
        <div class="selectTitle" style="height: 5%;">
            Instance Test
        </div>
        <div class="testpanel">
            <div class="testpanelView">
                <el-table ref="multipleTable" :data="exampleList" style="width: 100%" height="100%"
                    @selection-change="handleSelectionChange"
                    :cell-class-name="getCellClassName">
                    <el-table-column type="selection" width="40" fixed>
                    </el-table-column>
                    <el-table-column type="expand" width="35px" fixed>
                        <template v-slot="scope">
                            <div v-if="scope.row[0].id!=-1" class="allcontainertest"
                                style="display:flex; flex-direction: column;align-items: flex-start;justify-content: flex-start">
                                <div class="imageContainertest">
                                    <div v-for="img in scope.row" :key="img.id" class="imageWrappertest">
                                        <el-tooltip class="item" effect="light" placement="top">
                                            <template v-slot:content>
                                                <img :src="img.src" :alt="img.overall_explanation" style="width: 200px; height: auto;">
                                            </template>
                                            <img class="thumbnail" :src="img.src" :alt="img.overall_explanation">
                                        </el-tooltip>
                                    </div>
                                </div>
                                <div class="videotexttest">
                                    <b> Script: </b>{{ scope.row[0].text }}
                                </div>

                                <!-- <div class="videotexttest">
                                    <b> Explanation: </b> {{ scope.row[0].overall_explanation }}
                                </div> -->
                            </div>
                        </template>
                    </el-table-column>
                    <el-table-column label="Video Name" fixed>
                        <template v-slot="scope">
                            {{ scope.row[0].name }}
                        </template>
                    </el-table-column>
                    <el-table-column prop="gt" width="10px" fixed>
                    </el-table-column>
                    <el-table-column label="GT" width="90px" fixed>
                        <template v-slot="scope">
                            {{ scope.row[0].gtmap }}
                        </template>
                    </el-table-column>

                    <el-table-column prop="prediction" width="10px" fixed>
                        </el-table-column>
                    <el-table-column label="Prediction" width="90px" class-name="wrap-cell-text" fixed>
                        <template v-slot="scope">
                            {{ scope.row[0].resultmap }}
                        </template>
                    </el-table-column>
                    <el-table-column prop="match" label="Test Result" class-name="wrap-cell-text" width="100px" fixed>
                        <!-- <template v-slot="scope">
                            {{ scope.row.match }}
                        </template> -->
                        <template v-slot="scope">
                            <span v-if="scope.row[0].match">
                                <font-awesome-icon :icon="['far', 'circle-check']" style="color: #96d35f;" />
                            </span>
                            <span v-else>
                                <font-awesome-icon :icon="['far', 'circle-xmark']" style="color: #ff6251;" />
                            </span>
                        </template>
                    </el-table-column>
                    <el-table-column label="" width="50">
                        <template v-slot="scope">
                            <el-icon class="delete-icon" @click="deleteRow(scope.$index, scope.row)">
                                <delete />
                            </el-icon>
                        </template>
                    </el-table-column>
                </el-table>
            </div>
        </div>
    </div>
</template>

<script setup>
import {
    ElTable,
    ElTableColumn,
    ElIcon,
    ElTooltip
} from 'element-plus'
import { Delete } from '@element-plus/icons-vue'
import { library } from '@fortawesome/fontawesome-svg-core'
import { FontAwesomeIcon } from '@fortawesome/vue-fontawesome'
import { faCircleCheck,faCircleXmark } from '@fortawesome/free-regular-svg-icons'

</script>


<script>
import requesthelp from "../service/request.js";

export default {
    props: ['selectTestData','example','retrieveInstance'],
    data() {
        return {
            multipleSelection: [],
            exampleList: [],
            exampleData: {},
            allTestData: [],
            allRetrieveData: [],
            gt_mappings: {
                "-1": "NEG",
                "0": "NEU",
                "1": "POS"
            },
            result_mappings:{
                "positive": "POS",
                "neutral": "NEU",
                "negative": "NEG"
            }
        };
    },
    mounted() {
        // this.getExample()
    },
    watch: {
        retrieveInstance(){
            this.getRetrieve()
        },
        selectTestData(v) {
            // for (let i = v.length - 1; i >= 0; i--) {
            //     if (!this.allTestData.includes(v[i])) {
            //         this.allTestData.unshift(v[i])
            //     }
            // }
            this.fetchData(v)
        },
        example(){
            this.getExample()
        }
    },
    components: {
        'font-awesome-icon': FontAwesomeIcon
    },
    created() {
        library.add(faCircleCheck,faCircleXmark)
    },
    methods: {
        async getRetrieve(){
            var results = await requesthelp.axiosPost("/load_extra_test_example_with_instances", {
                video_list: this.retrieveInstance
            })
            console.log("retrieve",results)
            this.fetchDataRetrieve(results["retrieved_video_list"])
        },  
        getCellClassName({ row, column}) {
            if ((column.property === 'gt')) {
                switch (row[0].gt) {
                    case 1:
                        return 'positive';
                    case 0:
                        return 'neutral';
                    case -1:
                        return 'negative';
                    default:
                        return '';
                }
            }
            if ((column.property === 'prediction')) {
                switch (row[0].resultmap) {
                    case 'POS':
                        return 'positive';
                    case 'NEU':
                        return 'neutral';
                    case 'NEG':
                        return 'negative';
                    default:
                        return '';
                }
            }
            return '';
        },
        getImageSrc(key, frameNumber) {
            // Dynamically require the image based on the key and frame number
            try {

                return require(`@/assets/extracted_frames/${key}/frame_${String(frameNumber).padStart(4, '0')}.jpg`);
            } catch (e) {
                console.error(e);
                return ''; // Or return a placeholder image path
            }
        },
        deleteRow(index) {
            this.exampleList.splice(index, 1);
        },
        toggleSelection(rows) {
            if (rows) {
                rows.forEach(row => {
                    this.$refs.multipleTable.toggleRowSelection(row);
                });
            } else {
                this.$refs.multipleTable.clearSelection();
            }
        },
        handleSelectionChange(val) {
            this.multipleSelection = val;
        },
        async getExample() {
            this.exampleData = this.example["initial_result"];
            this.fetchData()
        },
        fetchDataRetrieve(data){
            for (let key in data) {
                if (!this.allRetrieveData.includes(key)) {
                    this.allRetrieveData.unshift(key)
                    // const prediction = data[key].prediction;
                    // const gt = data[key].gt
                    // this.exampleList.unshift({
                    //     name: key,
                    //     result: prediction,
                    //     resultmap: this.gt_mappings[prediction],
                    //     gt: gt,
                    //     gtmap: this.gt_mappings[gt.toString()],
                    //     match: this.gt_mappings[gt.toString()]=== this.gt_mappings[prediction]
                    // })

                    const mydata = data[key]
                    const imageGroup = []
                    // if (mydata){
                    for (let i = 1; i <= Number(mydata.frame_num); i++) {
                        const src = this.getImageSrc(key, i);
                        // console.log(src,"?")
                        const id = key + i.toString()
                        // const overall_explanation = mydata.overall_explanation
                        const text = data[key].script
                        const prediction = data[key].prediction;
                        const gt = data[key].gt
                        imageGroup.push({
                            id: id,
                            src: src,
                            // overall_explanation: overall_explanation,
                            text: text,
                            gt: gt,
                            name: key,
                            gtmap: this.gt_mappings[gt.toString()],
                            match: data[key].test_result,
                            result: prediction,
                            resultmap: this.gt_mappings[prediction],
                        })
                    }
                    if (imageGroup.length >= 1) {
                        // console.log(key,data);
                        this.exampleList.unshift(imageGroup)
                    } else {
                        const src = require(`@/assets/extracted_frames/noframe.png`);
                        const id = key + "0"
                        const overall_explanation = mydata.overall_explanation
                        const text = mydata.script
                        const prediction = data[key].prediction;
                        const gt = data[key].gt
                        imageGroup.push({
                            id: id,
                            src: src,
                            overall_explanation: overall_explanation,
                            text: text,
                            gt: gt,
                            name: key,
                            gtmap: this.gt_mappings[gt.toString()],
                            match: this.gt_mappings[gt.toString()]=== this.gt_mappings[prediction],
                            result: prediction,
                            resultmap: this.result_mappings[prediction],
                        })
                        this.exampleList.unshift(imageGroup)
                    }
                    // }else{
                    //     const src = require(`@/assets/extracted_frames/noframe.png`);
                    //     const id = -1
                    //     const overall_explanation = -1
                    //     const text = -1
                    //     const prediction = data[key].prediction;
                    //     const gt = data[key].gt
                    //     imageGroup.push({
                    //         id: id,
                    //         src: src,
                    //         overall_explanation: overall_explanation,
                    //         text: text,
                    //         gt: gt,
                    //         name: key,
                    //         gtmap: this.gt_mappings[gt.toString()],
                    //         match: this.gt_mappings[gt.toString()]=== this.gt_mappings[prediction],
                    //         result: prediction,
                    //         resultmap: this.result_mappings[prediction],
                    //     })
                    //     this.exampleList.unshift(imageGroup)
                    // }
                }
            }
        },
        fetchData(data) {
            // this.exampleList = []
            for (let i in data) {
                if (!this.allTestData.includes(data[i])) {
                    this.allTestData.unshift(data[i])
                    const key = data[i]
                    const mydata = this.exampleData[key]
                    // const mydata = this.exampleData[data[i]]
                    // const prediction = mydata.prediction.overall.toLowerCase();
                    // const gt = mydata.gt
                    // this.exampleList.unshift({
                    //     name: data[i],
                    //     result: prediction,
                    //     resultmap: this.result_mappings[prediction],
                    //     gt: gt,
                    //     gtmap: this.gt_mappings[gt.toString()],
                    //     match: this.gt_mappings[gt.toString()]=== this.result_mappings[prediction]
                    // })
                    const imageGroup = []
                    for (let i = 1; i <= Number(mydata.frame_num); i++) {
                        const src = this.getImageSrc(key, i);
                        // console.log(src,"?")
                        const id = key + i.toString()
                        const overall_explanation = mydata.overall_explanation
                        const text = mydata.script
                        const prediction = mydata.prediction.overall.toLowerCase();
                        const gt = mydata.gt
                        imageGroup.push({
                            id: id,
                            src: src,
                            overall_explanation: overall_explanation,
                            text: text,
                            gt: gt,
                            name: key,
                            gtmap: this.gt_mappings[gt.toString()],
                            match: this.gt_mappings[gt.toString()]=== this.result_mappings[prediction],
                            result: prediction,
                            resultmap: this.result_mappings[prediction],
                        })
                    }
                    if (imageGroup.length >= 1) {
                        // console.log(key,data);
                        this.exampleList.unshift(imageGroup)
                    } else {
                        const src = require(`@/assets/extracted_frames/noframe.png`);
                        const id = key + "0"
                        const overall_explanation = mydata.overall_explanation
                        const text = mydata.script
                        const prediction = mydata.prediction.overall.toLowerCase();
                        const gt = mydata.gt
                        imageGroup.push({
                            id: id,
                            src: src,
                            overall_explanation: overall_explanation,
                            text: text,
                            gt: gt,
                            name: key,
                            gtmap: this.gt_mappings[gt.toString()],
                            match: this.gt_mappings[gt.toString()]=== this.result_mappings[prediction],
                            result: prediction,
                            resultmap: this.result_mappings[prediction],
                        })
                        this.exampleList.unshift(imageGroup)
                    }
                }
            }
        },
    }

};
</script>

<style>
.testpanelView {
    /* overflow-y: auto;  */
    height: 98%;
}

.testpanel {
    height: 95%;
}

.testpanelContainer {
    height: 32vh;
    width: 100%;
}

th,
td {
    border: 0px solid #ddd;
}

.el-table .el-table__cell {
    padding: 3px 0 !important;
}

.wrap-cell-text {
    white-space: normal !important;
    word-break: break-word !important;
}

.delete-icon {
    cursor: pointer;
    color: #F56C6C; /* 举例颜色，可以根据需要修改 */
}
.delete-icon:hover {
    color: #ff7875; /* 鼠标悬停时的颜色 */
}

.videotexttest {
    height: auto;
    width: 100%;
    margin-left: 10px;
    margin-bottom: 5px;
    line-height: 18px;
}
.allcontainertest {
    height: auto;
    width: 100%;
}

.imageWrappertest{
    position: relative;
    margin-right: 2px;

}

.imageContainertest {
    display: flex;
    flex-wrap: nowrap;
    overflow-x: auto;
    overflow-y: hidden;
    gap: 2px;
    height: auto;
    width: 100%;
    margin-left: 10px;
    margin-bottom: 5px;
}


.imageContainertest img {
    flex: 0 0 auto;
    height: auto;
    max-height: 30px;
    max-width: auto;
    padding-top: 0px;
}

</style>
