We have selected multiple videos to make inferences and use as points of comparison from the [StreetAware Dataset](https://drive.google.com/drive/folders/1BPtiIF8gBOoZANAGkwDjJUYakpCUYHM1?usp=sharing).
<br>
<br>
<b> Chosen Videos: </b>
<ul>
  <li>chase_1 -> sensor_1 -> left_half.mp4 (45 minutes)</li>
  <li>chase_3 -> sensor_1 -> left_half.mp4 (45 minutes)</li>
  <li>park</li>
  <li>park</li>
  <li>dumbo_3 -> sensor_2 -> left_half.mp4 (30 minutes)</li>
</ul>
<br>
Each dataset contains 2 files: YOLO+RAM.csv and Yolo_Counts.csv

<br>
<br>

<h3>Below is how we identified different location and videos </h3>

<h4>Locations and Video IDs</h4>
<p>We have assigned the following Location IDs to specific locations:</p>
<ul>
  <li>Park: 1</li>
  <li>Chase: 2</li>
  <li>Dumbo: 3</li>
</ul>

<p>We have also assigned the following Video IDs to specific locations of cameras:</p>
<ul>
  <li>Park1: 1</li>
  <li>Chase1: 2</li>
  <li>Chase3: 3</li>
  <li>Dumbo3: 4</li>
</ul>

<h4>Aggregation IDs (aggID)</h4>
<p>To create a unique identifier <code>aggID</code> for each observation, we use the formula:</p>
<b>aggID = locationID * 1000000 + videoID * 100000 + image#</b>

<p>Here is an example to illustrate:</p>
<p>If we want to get <b>aggID</b> for Chase, Video 3 (Chase3), Image #84: 3300084</p>
<p>This ensures that each observation has a unique aggID based on its location, video, and image number.</p>
