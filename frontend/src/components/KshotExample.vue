<template>
	<div class="kshotexampleContainer view">
        <div style="width: 100%; height: 15%; display: flex; align-items: flex-start; justify-content: flex-start">
    
            <div class="selectTitle" style="width: 95%;">
                K-shot Example List
            </div>
            <!-- <el-button style="margin-right: 5px; margin-top: 5px" type="info" @click="UploadItems()" :icon="DocumentAdd" round
                plain>Import </el-button> -->
        </div>
		<div class="example">
			<div class="exampleView">
                <el-table
                    ref="multipleTable"
                    :data="exampleList"
                    style="width: 100%" height="100%"
                    @selection-change="handleSelectionChange"
                    :cell-class-name="getCellClassName">
                    <el-table-column
                        type="selection"
                        width="40" fixed>
                    </el-table-column>
                    <el-table-column
                        label="Video Name"
                        fixed>
                        <template v-slot="scope">
                            {{ scope.row.name }}
                        </template>
                    </el-table-column>
                    <el-table-column prop="gt" width="10px" fixed>
                        </el-table-column>
                    <el-table-column
                         label="GT"
                        width="250px" fixed>
                        <template v-slot="scope">
                            {{ scope.row.gtmap }}
                        </template>
                    </el-table-column>
                    <!-- <el-table-column
                        label="Label"
                        class-name="wrap-cell-text" fixed>
                        <template v-slot="scope">
                            {{ scope.row.label }}
                        </template>
                    </el-table-column> -->
                </el-table>
            </div>
		</div>
	</div>
</template>

<script setup>
import {
    ElTable,
    ElTableColumn
} from 'element-plus'
// import { DocumentAdd } from '@element-plus/icons-vue'
</script>


<script>
// import requesthelp from "../service/request.js";

export default {
    props: ['kshotExampleData','example'],
    data() {
        return {
            multipleSelection: [],
            exampleList: [],
            exampleData: {},
            gt_mappings: {
                "-1": "NEG",
                "0": "NEU",
                "1": "POS"
            },
            recievedList: [],
            uploaddata: []
        };
    },
    mounted(){
        // this.getExample()
    },
    watch:{
        kshotExampleData(v){
            for (let i = v.length - 1; i >= 0; i--) {
                if (!this.recievedList.includes(v[i])) {
                    this.recievedList.unshift(v[i])
                }
            }
            console.log("save kshot",this.kshotExampleData, this.recievedList)
            this.fetchData()
        },
        example(){
            this.getExample()
        }
    },
    methods:{
        UploadItems() {
            this.$emit('kshotdata', this.uploaddata);
        },
        getCellClassName({ row, column}) {
            if ((column.property === 'gt')) {
                switch (row.gt) {
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
            return '';
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
            // console.log(val)
            var uploaddata = []
            for (let i=0; i<this.multipleSelection.length; i++){
                this.exampleData[this.multipleSelection[i].name].reasoning.explanation = this.multipleSelection[i].reasoning
                uploaddata[this.multipleSelection[i].name] = this.exampleData[this.multipleSelection[i].name]
            }
            this.uploaddata = uploaddata
            this.$emit('kshotdata', this.uploaddata);
            console.log(this.uploaddata)
        },
        async getExample(){
            // var results = await requesthelp.axiosGet("/load_k_shot_example")
            // console.log("kshot", results)
            // this.exampleData = results["k_shot_example_dict"];
            this.exampleData = this.example["k_shot_example_dict"];
            // this.fetchData()
        },
        fetchData(){
            // console.log("instance", this.exampleData)
            this.exampleList = []
            for (let key in this.recievedList){
                const data = this.exampleData[this.recievedList[key].name]
                const label = data.label.toFixed(2)
                const gt = data.gt
                this.exampleList.push({
                    name: this.recievedList[key].name,
                    label: label,
                    gt: gt,
                    gtmap: this.gt_mappings[gt.toString()],
                    reasoning: this.recievedList[key].reasoning
                })
            }
        },
    }
    
};
</script>

<style>

.exampleView {
    height: 99%;
}
.example {
    height: 85%;
}
.kshotexampleContainer{
    height: 17vh;
    width: 100%;
}

th, td{
    border: 0px solid #ddd;
}

.el-table .el-table__cell{
    padding: 3px 0 !important;
}

.wrap-cell-text {
    white-space: normal !important;
    word-break: break-word !important;
}


</style>
