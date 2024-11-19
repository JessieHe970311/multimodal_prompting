<template>
    <div class="reasoningContainer view">
        <!-- <div class="selectTitle" style="height: 5%">Instance/Group Level</div> -->
        <div class="reasoningView" style="height: 100%">
            <div class="switchbutton" style="display: flex; justify-content: space-between; height: 5%; padding-bottom: 15px">
                <div>
                    <span v-if="isTestMode">K-shot Example</span>
                    <span v-else><b>K-shot Example</b></span>
                    <el-switch class="tableswith" v-model="isTestMode" active-color="#13ce66" inactive-color="#c9c9c9"
                        @change="handleSwitchChange">
                    </el-switch>
                    <span v-if="isTestMode"><b>Validation Instance</b></span>
                    <span v-else>Validation Instance</span>
                </div>
                <div>
                    <el-button style="margin-right: 5px" type="info" @click="RetrieveItems()" round plain>
                        <font-awesome-icon :icon="['fas', 'inbox']" />&nbsp; Retrieve </el-button>
                    <el-button style="margin-right: 5px" type="info" @click="GenerateItems()" round plain>
                        <font-awesome-icon icon="wand-magic-sparkles" /> &nbsp;Generate </el-button>
                    <el-button style="margin-right: 5px" type="info" @click="SaveItems()" :icon="CirclePlusFilled" round
                        plain> Save </el-button>
                    <!-- <el-button style="margin-right: 5px" type="info" @click="SaveItems()" :icon="CirclePlusFilled" round plain> Save </el-button> -->
                </div>
            </div>
            <div class="tablecontainer" style="height: 90%">
                <div v-if="isTestMode" class="mytable">
                    <el-table :data="images" style="width: 100%" height="100%" :cell-class-name="getCellClassName"
                        @selection-change="handleTestSelectionChange">
                        <el-table-column type="selection" width="40" fixed>
                        </el-table-column>
                        <el-table-column prop="gt" width="10px" fixed>
                        </el-table-column>
                        <el-table-column label="GT" width="45px" fixed>
                            <template v-slot="scope">
                                <!-- <div> -->
                                {{ scope.row[0].gtmap }}
                                <!-- </div> -->
                            </template>
                        </el-table-column>
                        <el-table-column prop="prediction" width="10px" fixed>
                        </el-table-column>
                        <el-table-column label="Pred" width="60px" fixed>
                            <template v-slot="scope">
                                <!-- <div> -->
                                {{ scope.row[0].prediction }}
                                <!-- </div> -->
                            </template>
                        </el-table-column>
                        <el-table-column prop="video" label="" fixed  width="45px">
                            <template v-slot="scope">
                                <el-tooltip popper-class="tooltip-width" effect="light" placement="top"
                                    :content="scope.row[0].videoUrl" open-delay="1000">
                                    <template v-slot:content>
                                        <video :src="scope.row[0].videoUrl" controls autoplay
                                            style="width: 300px; height: auto;"></video>
                                    </template>
                                    <el-button circle>
                                        <font-awesome-icon icon="play" style="color: #6c84ac;margin-left:2px" />
                                    </el-button>
                                </el-tooltip>
                            </template>
                        </el-table-column>
                        <el-table-column prop="rawData" label="Raw Data" fixed width="300px">
                            <template v-slot="scope">
                                <div class="allcontainer"
                                    style="display:flex; flex-direction: column;align-items: flex-start;justify-content: flex-start">
                                    <div class="imageContainer">
                                        <div v-for="img in scope.row" :key="img.id" class="imageWrapper">
                                            <el-tooltip class="item" effect="light" placement="top">
                                                <template v-slot:content>
                                                    <img :src="img.src" :alt="img.overall_explanation"
                                                        style="width: 200px; height: auto;">
                                                </template>
                                                <img class="thumbnail" :src="img.src" :alt="img.overall_explanation">
                                            </el-tooltip>
                                            <!-- <img class="thumbnail" :src="img.src" :alt="img.overall_explanation"
                                                @mouseenter="showPreview($event, img)" @mouseleave="hidePreview(img)" /> -->
                                            <!-- <img class="preview" :src="img.src" :alt="img.overall_explanation"
                                                :ref="`preview-${img.id}`" /> -->
                                        </div>
                                    </div>
                                    <div class="videotext">
                                        {{ scope.row[0].text }}
                                    </div>
                                </div>
                            </template>
                        </el-table-column>
                        <el-table-column prop="explanation" label="Explanation" fixed>
                            <template v-slot="scope">
                                <!-- {{ scope.row[0].overall_explanation }} -->
                                <!-- <el-tooltip popper-class="tooltip-width" effect="light" placement="top" -->
                                <!-- :content="scope.row[0].overall_explanation" raw-content> -->
                                <div class="wrap-text" style="padding: 10px" v-html="scope.row[0].overall_explanation">
                                </div>
                                <!-- </el-tooltip> -->
                            </template>
                        </el-table-column>
                    </el-table>
                </div>

                <div v-else class="mytable">
                    <el-table :data="kshotimages" style="width: 100%" height="100%" :cell-class-name="getCellClassName"
                        @selection-change="handleKShotSelectionChange">
                        <el-table-column type="selection" width="40" fixed>
                        </el-table-column>
                        <el-table-column prop="gt" width="10px" fixed>
                        </el-table-column>
                        <el-table-column label="GT" width="50px" fixed align="center">
                            <template v-slot="scope">
                                {{ scope.row[0].gtmap }}
                            </template>
                        </el-table-column>
                        <el-table-column prop="video" label="" fixed  width="45px">
                            <template v-slot="scope">
                                <el-tooltip popper-class="tooltip-width" effect="light" placement="top"
                                    :content="scope.row[0].videoUrl" open-delay="1000">
                                    <template v-slot:content>
                                        <video :src="scope.row[0].videoUrl" controls autoplay
                                            style="width: 300px; height: auto;"></video>
                                    </template>
                                    <el-button circle>
                                        <font-awesome-icon icon="play" style="color: #6c84ac;margin-left:2px" />
                                    </el-button>
                                </el-tooltip>
                            </template>
                        </el-table-column>
                        <el-table-column prop="rawData" label="Raw Data" fixed width="250px">
                            <template v-slot="scope">
                                <div class="allcontainer"
                                    style="display:flex; flex-direction: column;align-items: flex-start;justify-content: center">
                                    <div class="imageContainer">
                                        <div v-for="img in scope.row" :key="img.id" class="imageWrapper">
                                            <!-- <img class="thumbnail" :src="img.src" :alt="img.overall_explanation"
                                                @mouseenter="showPreview($event, img)" @mouseleave="hidePreview(img)" />
                                            <img class="preview" :src="img.src" :alt="img.overall_explanation"
                                                :ref="`preview-${img.id}`" /> -->
                                            <el-tooltip class="item" effect="light" placement="top">
                                                <template v-slot:content>
                                                    <img :src="img.src" :alt="img.overall_explanation"
                                                        style="width: 200px; height: auto;">
                                                </template>
                                                <img class="thumbnail" :src="img.src" :alt="img.overall_explanation">
                                            </el-tooltip>
                                        </div>
                                    </div>
                                    <div class="videotext">
                                        {{ scope.row[0].text }}
                                    </div>
                                </div>
                            </template>
                        </el-table-column>
                        <el-table-column prop="reasoning" label="Reasoning" fixed>
                            <template v-slot="scope">
                                <div class="wrap-text" style="padding: 10px" v-html="scope.row[0].overall_explanation"
                                    v-if="!scope.row[0].editing">
                                </div>

                                <el-input v-else type="textarea" v-model="scope.row[0].overall_explanation">
                                </el-input>
                                <el-button link @click="editItem(scope.row[0])"
                                    :icon="scope.row[0].editing ? Check : Edit">
                                </el-button>

                            </template>
                        </el-table-column>
                    </el-table>
                </div>
            </div>
        </div>
    </div>
</template>

<script setup>
import {
    ElTable,
    ElTableColumn,
    ElSwitch,
    ElButton,
    ElInput,
    ElTooltip,
} from 'element-plus'
import { CirclePlusFilled } from '@element-plus/icons-vue'
import { Edit, Check } from '@element-plus/icons-vue'
import { library } from '@fortawesome/fontawesome-svg-core'
import { FontAwesomeIcon } from '@fortawesome/vue-fontawesome'
import { faWandMagicSparkles, faPlay } from '@fortawesome/free-solid-svg-icons'
import { faInbox } from '@fortawesome/free-solid-svg-icons'
</script>

<script>
// import myData from '@/assets/data_dict_0212_part.json';

// import requesthelp from "../service/request.js";

export default {
    props: ['testInstance', 'allInstancedata', 'kshotInstance'],
    data() {
        return {
            images: new Array(),
            kshotimages: new Array(),
            instanceData: new Array(),
            isTestMode: true,
            selectedInstance: new Array(),
            selectedTest: new Array(),
            selectedKshot: new Array(),
            gt_mappings: {
                "-1": "NEG",
                "0": "NEU",
                "1": "POS"
            },
            pre_mappings: {
                "negative": "NEG",
                "neutral": "NEU",
                "positive": "POS"
            }
        };
    },
    components: {
        'font-awesome-icon': FontAwesomeIcon
    },
    created() {
        library.add(faWandMagicSparkles, faInbox,faPlay)
    },
    mounted() {
        // this.getInstance()
        // this.getTest()
    },
    watch: {
        testInstance(v) {
            this.selectedInstance = v;
            if (this.isTestMode) {
                this.fetchData()
            }
        },
        allInstancedata() {
            this.getInstance()
        },
        kshotInstance() {
            this.getTest()
        }
    },
    methods: {
        editItem(item) {
            item.editing = !item.editing;
        },
        GenerateItems() {
            if (this.isTestMode) {
                this.$emit('selectkshotdata', this.selectedTest);
            } else {
                alert('Cannot generate! Not Test Mode');
            }
        },
        SaveItems() {
            if (this.isTestMode) {
                this.$emit('selecttestdata', this.selectedTest);
            } else {
                // alert('Cannot save! Not Test Mode');
                this.$emit('selectkshotexampledata', this.selectedKshot);
            }
        },
        RetrieveItems() {
            if (this.isTestMode) {
                this.$emit('retrievetestdata', this.selectedTest);
            }
        },
        handleTestSelectionChange(selectedItems) {
            this.selectedTest = []
            for (let i = 0; i < selectedItems.length; i++) {
                this.selectedTest.push(selectedItems[i][0].videoname)
            }
        },
        handleKShotSelectionChange(selectedItems) {
            this.selectedKshot = []
            // console.log(selectedItems)
            for (let i = 0; i < selectedItems.length; i++) {
                this.selectedKshot.push({
                    name: selectedItems[i][0].videoname,
                    reasoning: selectedItems[i][0].overall_explanation
                })
            }
        },
        handleSwitchChange(v) {
            this.isTestMode = v
            // this.selectedTest = []
            // this.selectedKshot = []
            // if (!v) {
            //     this.getInstance()
            // } else {
            // this.getTest()
            // }
        },
        getCellClassName({ row, column }) {
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
                switch (row[0].prediction) {
                    case "POS":
                        return 'positive';
                    case "NEU":
                        return 'neutral';
                    case "NEG":
                        return 'negative';
                    default:
                        return '';
                }
            }
            return '';
        },
        // 真正的test
        fetchData() {
            this.images = []
            for (let key in this.instanceData) {
                if (this.selectedInstance.includes(key) || this.selectedInstance.length === 0) {
                    const data = this.instanceData[key]
                    const imageGroup = []
                    for (let i = 1; i <= Number(data.frame_num); i++) {
                        const src = this.getImageSrc(key, i);
                        // console.log(src,"?")
                        const id = key + i.toString()
                        const overall_explanation = this.highlightTextSnippets(data.overall_explanation, data.reasoning.language_cues, data.reasoning.visual_cues)
                        const text = data.script
                        const gt = data.gt
                        const prediction = data.prediction.overall
                        imageGroup.push({
                            id: id,
                            src: src,
                            overall_explanation: overall_explanation,
                            text: text,
                            gt: data.gt,
                            videoname: key,
                            gtmap: this.gt_mappings[gt.toString()],
                            prediction: this.pre_mappings[prediction.toLowerCase()],
                            videoUrl: require(`@/assets/segmented_videos_refine/${key}.mp4`)
                        })
                    }
                    if (imageGroup.length >= 1) {
                        // console.log(key,data);
                        this.images.push(imageGroup)
                    } else {
                        const src = require(`@/assets/extracted_frames/noframe.png`);
                        const id = key + "0"
                        const overall_explanation = this.highlightTextSnippets(data.overall_explanation, data.reasoning.language_cues, data.reasoning.visual_cues)
                        const text = data.script
                        const gt = data.gt
                        const prediction = data.prediction.overall
                        imageGroup.push({
                            id: id,
                            src: src,
                            overall_explanation: overall_explanation,
                            text: text,
                            gt: data.gt,
                            videoname: key,
                            gtmap: this.gt_mappings[gt.toString()],
                            prediction: this.pre_mappings[prediction.toLowerCase()],
                            videoUrl: require(`@/assets/segmented_videos_refine/${key}.mp4`)
                        })
                        this.images.push(imageGroup)
                    }
                }
            }
        },
        // k-shot!!
        fetchTestData() {
            // console.log("instance", this.instanceData, this.selectedInstance)
            this.kshotimages = []
            for (let key in this.instanceData) {
                // if (this.selectedInstance.includes(key)) {
                const data = this.instanceData[key]
                const imageGroup = []
                for (let i = 1; i <= Number(data.frame_num); i++) {
                    const src = this.getImageSrc(key, i);
                    // console.log(src,"?")
                    const id = key + i.toString()
                    // const overall_explanation = this.highlightTextSnippets(data.reasoning.explanation, data.reasoning.language_cues, data.reasoning.visual_cues)
                    const overall_explanation = data.reasoning.explanation
                    const text = data.script
                    const gt = data.gt
                    imageGroup.push({
                        id: id,
                        src: src,
                        overall_explanation: overall_explanation,
                        text: text,
                        gt: gt,
                        editing: false,
                        videoname: key,
                        gtmap: this.gt_mappings[gt.toString()],
                        videoUrl: require(`@/assets/segmented_videos_refine/${key}.mp4`)
                    })
                }
                if (imageGroup.length >= 1) {
                    // console.log(key,data);
                    this.kshotimages.push(imageGroup)
                } else {
                    const src = require(`@/assets/extracted_frames/noframe.png`);
                    const id = key + "0"
                    // const overall_explanation = this.highlightTextSnippets(data.reasoning.explanation, data.reasoning.language_cues, data.reasoning.visual_cues)
                    const overall_explanation = data.reasoning.explanation
                    const text = data.script
                    const gt = data.gt
                    imageGroup.push({
                        id: id,
                        src: src,
                        overall_explanation: overall_explanation,
                        text: text,
                        gt: gt,
                        editing: false,
                        videoname: key,
                        gtmap: this.gt_mappings[gt.toString()],
                        videoUrl: require(`@/assets/segmented_videos_refine/${key}.mp4`)
                    })
                    this.kshotimages.push(imageGroup)
                }

                // }
            }
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

        async getInstance() {
            // var results = await requesthelp.axiosGet("/load_initial_prompt_and_result")
            // this.instanceData = results["initial_result"];
            this.instanceData = this.allInstancedata["initial_result"];
            this.fetchData()
        },

        async getTest() {
            // var results = await requesthelp.axiosGet("/load_k_shot_example")
            // this.instanceData = results["k_shot_example_dict"];
            this.instanceData = this.kshotInstance["k_shot_example_dict"];
            this.fetchTestData()
        },

        showPreview(event, img) {
            // 查找预览图元素
            const preview = this.$refs[`preview-${img.id}`][0];

            // 获取鼠标位置
            const mouseX = event.clientX;
            const mouseY = event.clientY;

            // 计算预览图可能的位置
            let previewX = mouseX;
            let previewY = mouseY - preview.offsetHeight - 15;

            // 获取窗口的宽度和高度
            const windowWidth = window.innerWidth;
            // const windowHeight = window.innerHeight;

            // 检查预览图是否会超出窗口的右侧
            if (previewX + preview.offsetWidth > windowWidth) {
                previewX = mouseX - preview.offsetWidth - 15;
            }

            // 检查预览图是否会超出窗口的顶部
            if (previewY < 0) {
                previewY = mouseY + 15;
            }

            // 设置预览图的位置
            preview.style.left = `${previewX}px`; // 横坐标
            preview.style.top = `${previewY}px`; // 纵坐标
            preview.style.display = 'block'; // 显示预览图
        },

        hidePreview(img) {
            // 查找预览图元素并隐藏
            const preview = this.$refs[`preview-${img.id}`][0];
            preview.style.display = 'none';
        },

        highlightTextSnippets(text, cues, vcues) {
            let highlightedText = text;
            // console.log(cues)
            // Go through each cue and replace it with a highlighted version
            const allCues = [...cues, ...vcues];
            allCues.forEach(cue => {
                const [cueText, sentiment] = cue;
                const regex = new RegExp(this.escapeRegExp(cueText), 'gi');
                const className = `highlight-${sentiment.toLowerCase()}`;
                highlightedText = highlightedText.replace(regex, `<span class="${className}">$&</span>`);
            });

            return highlightedText;
        },
        escapeRegExp(string) {
            return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        }
    }
}

</script>

<style>
.reasoningContainer {
    height: 45vh;
}

.mytable {
    width: 100%;
    height: 100%;
    max-height: 100%;
    /* Adjust height based on your layout */
}

.reasoningView {
    height: 100%;
    width: 100%;
}

.el-table__header-wrapper {
    background-color: #f2f2f2;
}

.el-table__body-wrapper {
    overflow-y: scroll;
}

.el-textarea__inner {
    height: 140px;
}

.tableswith {
    height: 100%;
}

.mytable .el-table .el-table__body .cell {
    height: 150px !important;
    padding: 10 !important;
    display: flex;
    align-items: center;
}

.tooltip-width {
    max-width: 600px;
    font-family: sans-serif;;
    color: #3b3b3b;
}

table {
    width: 100%;
    border-collapse: collapse;
    box-sizing: border-box;
}

th,
td {
    border: 1px solid #ddd;
    text-align: left;
    word-wrap: break-word;
    white-space: normal;
}

thead {
    background-color: #d2d2d2;
}


.imageContainer {
    display: flex;
    flex-wrap: nowrap;
    overflow-x: auto;
    overflow-y: hidden;
    gap: 2px;
    height: 100px;
    width: 100%;
    margin-left: 10px;
}

.allcontainer {
    height: 100px;
    width: 100%;
}


.wrap-text {
    display: -webkit-box;
    -webkit-box-orient: vertical;
    -webkit-line-clamp: 5;
    /* 设置最多显示的行数为 4 */
    overflow: auto;
    text-overflow: ellipsis;
    word-break: break-word;
    font-family: sans-serif;
}

.videotext {
    height: 120px;
    width: 100%;
    font-family: sans-serif;
    margin-left: 10px;
    line-height: 18px;
    overflow-y: auto;
}

.imageContainer img {
    flex: 0 0 auto;
    height: auto;
    max-height: 30px;
    max-width: auto;
    padding-top: 0px;
}

.imageWrapper {
    position: relative;
    margin-right: 2px;
    /* 为图片之间提供间隙 */
}

.thumbnail {
    display: block;
    max-height: 150px;
    /* 或者您希望的高度 */
    padding-top: 50px;
}

.preview {
    display: none;
    /* 默认不显示预览图 */
    position: fixed;
    /* 相对于视口固定定位 */
    z-index: 100;
    /* 确保预览图在最上方 */
    border: 3px solid white;
    box-shadow: 0px 8px 16px rgba(0, 0, 0, 0.2);
}

.imageWrapper:hover .preview {
    display: block;
    /* 当鼠标悬停时显示预览图 */
    max-width: 100%;
    /* 限制预览图的最大宽度 */
    max-height: 150px;
    /* 限制预览图的最大高度 */
}

.highlight-positive {
    background-color: #CB827C;
}

.highlight-neutral {
    background-color: #F4EDE1;
}

.highlight-negative {
    background-color: #8A9DBC;
}



/* colorarray: ["#A1AECE", "#F5EEE3", "#CDA2A2"], */

.el-table .positive {
    background: #CB827C !important;
}

.el-table .neutral {
    background: #F4EDE1 !important;
}

.el-table .negative {
    background: #8A9DBC !important;
}
</style>