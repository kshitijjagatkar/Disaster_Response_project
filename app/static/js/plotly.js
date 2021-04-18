

//------------------------------------Top Countries bar chart ---------------------------------------------------//
var countries = ['United States', 'India', 'United Kingdom', 'Germany', 'Australia',
       'France', 'Canada', 'Netherlands', 'Spain', 'Switzerland'];
var c_value = [32.81, 21.34, 15.81,  7.91,  4.74,  4.35,  4.35,  3.16,  2.77, 2.77];

var trace1 = {
  x:countries,
  y:c_value,
  type:'bar'
};

var layout1 = {
  xaxis: {title:{text: "Countries" },fixedrange:true},
  yaxis: {title:{text:"Percentage"}}
  };

var data1 = [trace1];
var config = {responsive: true,scrollZoom: true,displayModeBar: true};
Plotly.newPlot("country_bar",data1,layout1,config);

//-----------------------------------------Gender of ds/ml Pie chart----------------------------------------------//
var gender = ['man','woman','other'];
var gender_values = [255,  53,   4];

var trace2 = {
  labels:gender,
  values:gender_values,
  type:'pie',
  textinfo:"label+percent"
};

var layout = {
  margin:"auto"
}

var data2 = [trace2];
var config = {responsive:true,scrollZoom: true,displayModeBar: true}

Plotly.newPlot("gender_pie",data2,layout,config);

//-----------------------Age of The Data Scientist / Machine Learning Specialist Bar chart-------------------------------------------------------//
var age = [26, 28, 25, 31, 24, 29, 27, 30, 23, 22, 38, 35, 32, 34, 33, 41, 21,
            36, 39, 40, 42, 46, 48, 45, 20, 63, 37, 50, 53, 54, 62, 44, 47, 49,
            55, 56, 61, 19, 18, 16, 15];
var age_count = [11.35,  8.87,  6.38,  6.03,  5.32,  5.32,  4.96,  4.96,  4.61,
        3.9 ,  3.55,  3.19,  3.19,  2.84,  2.84,  2.84,  2.48,  1.77,
        1.42,  1.42,  1.42,  1.06,  1.06,  1.06,  0.71,  0.71,  0.71,
        0.71,  0.71,  0.71,  0.35,  0.35,  0.35,  0.35,  0.35,  0.35,
        0.35,  0.35,  0.35,  0.35,  0.35];
var trace3 = {
  x:age,
  y:age_count,
  type:'bar'
};

var layout2 = {
  title:{text:"Age of The Data Scientist/Machine Learning Specialist",autosize:true},
  xaxis: {title:{text: "Age"}},
  yaxis: {title:{text:"Percentage"}}
};
var data3 = [trace3];
var config = {responsive: true,scrollZoom: true,displayModeBar: true};
Plotly.newPlot("Age_bar",data3,layout2,config);

//--------------------Highest level of education for data scientist/machine learning specialist Bar chart---------------------------------------------//
var formal_ed = ['Master’s degree (M.A., M.S., M.Eng., MBA, etc.)',
       'Bachelor’s degree (B.A., B.S., B.Eng., etc.)',
       'Other doctoral degree (Ph.D., Ed.D., etc.)',
       'Some college/university study without earning a degree',
       'Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)',
       'Professional degree (JD, MD, etc.)',
       'Associate degree (A.A., A.S., etc.)',
       'I never completed any formal education', 'Primary/elementary school'];
var formal_ed_count = [45.33, 30.77, 17.03,  2.47,  2.2 ,  0.82,  0.55,  0.55,  0.27];

var trace4 = {
  x:formal_ed,
  y:formal_ed_count,
  type:'bar'
};

var layout3 = {
  title: {text:"Highest level of education"},
  xaxis: {title:{text:"hover/tap over fig to see the education type",
                 font:{color:"lightgrey"}},fixedrange:true,
          ticks:'',
          showticklabels:false},
  yaxis: {title:{text:"Percentage %"}}
};
var data4 = [trace4];
var config = {responsive: true,scrollZoom: true,displayModeBar: true};
Plotly.newPlot("formal_edu",data4,layout3,config);

//------------------------------------------------------Major Field of Study BAR CHART----------------------------------------------------------------------------------//

var major_ed = ['Computer science, computer engineering, or software engineering',
       'Mathematics or statistics',
       'A natural science (such as biology, chemistry, physics, etc.)',
       'Another engineering discipline (such as civil, electrical, mechanical, etc.)',
       'A business discipline (such as accounting, finance, marketing, etc.)',
       'A social science (such as anthropology, psychology, political science, etc.)',
       'Information systems, information technology, or system administration',
       'A humanities discipline (such as literature, history, philosophy, etc.)',
       'A health science (such as nursing, pharmacy, radiology, etc.)',
       'Fine arts or performing arts (such as graphic design, music, studio art, etc.)'];
var major_ed_count = [33.92, 23.98, 14.04, 14.04,  3.8 ,  3.8 ,  2.63,  2.05,  1.46,
        0.29];

var trace5 = {
  x:major_ed,
  y:major_ed_count,
  type:'bar'
};

var layout4 = {
  title: {text:"Major Field of Study"},
  xaxis: {title:{text:"hover over to see the Major type",
                 font:{color:"lightgrey"}},fixedrange:true,
          ticks:'',
          showticklabels:false},
  yaxis: {title:{text:"Percentage %"}}
};
var data5 = [trace5];
var config = {responsive: true,scrollZoom: true,displayModeBar: true};
Plotly.newPlot("major_edu",data5,layout4,config);

//-------------------------------------------------------------The Industrial Analysis---------------------------------------------//
//-----------------------------------------------------------Top Languages,Tools & Software---------------------------------------------//


var langs = ['Python', 'SQL', 'R', 'Bash/Shell/PowerShell', 'HTML/CSS', 'JavaScript',
       'C++', 'Java', 'C', 'C#'];
var langs_values = [27.15, 16.13, 12.4 ,  9.02,  7.29,  6.76,  6.68,  6.68,  5.2 ,
        2.69];

var trace6 = {
  labels:langs,
  values:langs_values,
  type:'pie',
  textinfo:"label+percent",
  automargin:true
};

var layout5 = {
  margin:"auto",
  title:{text:"Top Languages,Tools & Software used in 2020",autosize:true},
};

var data6 = [trace6];
var config = {responsive:true,scrollZoom: true,displayModeBar: true}

Plotly.newPlot("lang_pie",data6,layout5,config);


//--------------------------------------------------coding experience-----------------------------------------------//

var yrs_code = ['5', '10', '6', '4', '7', '8', '15', '3', '2', '12', '20', '9', '11',
       '13', '16', '30', '14', '22', '18', '25', '<1', '17', '21', '38', '24',
       '1', '50', '44', '>50', '32', '40', '35', '26', '36', '19', '28', '27',
       '37', '33'];
var yrs_code_count = [11.33,  9.39,  9.12,  7.73,  6.35,  6.08,  5.52,  5.52,  4.14,
        3.59,  3.59,  3.04,  2.49,  2.21,  1.66,  1.66,  1.66,  1.66,
        1.38,  1.38,  1.1 ,  1.1 ,  1.1 ,  0.83,  0.83,  0.55,  0.55,
        0.55,  0.55,  0.55,  0.55,  0.28,  0.28,  0.28,  0.28,  0.28,
        0.28,  0.28,  0.28];

var trace7 = {
  x:yrs_code,
  y:yrs_code_count,
  type:'bar'
};

var layout6 = {
  title: {text:"Overall coding experience"},
  xaxis: {title:{text:"Coding experience in years"}},
  yaxis: {title:{text:"Percentage %"}}
};
var data7 = [trace7];
var config = {responsive: true,scrollZoom: true,displayModeBar: true};
Plotly.newPlot("yrs_code",data7,layout6,config);


//------------------------------------------------------------Job satisfaction-------------------------------------------------------//

var job_satisfaction = ['Very satisfied', 'Slightly satisfied', 'Slightly dissatisfied',
       'Neither satisfied nor dissatisfied', 'Very dissatisfied'];
var job_satisfaction_values = [42.63, 27.27, 12.23,  9.72,  8.15];

var trace8 = {
  labels:job_satisfaction,
  values:job_satisfaction_values,
  type:'pie',
  textinfo:"label+percent",
  automargin:true
};

var layout7 = {
  margin:"auto",
  title:{text:"Job satisfaction",autosize:true},
};

var data8 = [trace8];
var config = {responsive:true,scrollZoom: true,displayModeBar: true}

Plotly.newPlot("job_sat",data8,layout7,config);



//-----------------------------------------------highest salaries---------------------------------------//

var sal = ['United States', 'United Kingdom', 'Australia', 'Canada', 'Switzerland',
       'Ireland', 'France', 'Germany', 'Austria', 'Netherlands'];
var sal_count = [147500.0,
 73031.5,
 106895.0,
 62018.0,
 109425.19,
 64859.0,
 62156.5,
 67021.0,
 55309.0,
 70264.0];

var trace9 = {
  x:sal,
  y:sal_count,
  type:'scatter'
};

var layout8 = {
  title: {text:"The salary of ds/ml sp."},
  xaxis: {title:{text:"Countries"},fixedrange:true},
  yaxis: {title:{text:"Avg Salary per year in $"}}
};
var data9 = [trace9];
var config = {responsive: true,scrollZoom: true,displayModeBar: true};
Plotly.newPlot("salary",data9,layout8,config);


//--------------------------------------------------------------------------------------------------------//

var sal_us = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51];

var sal_us_count = [2000000., 1680000., 1560000.,  490000.,  325000.,  260000.,
        250000.,  250000.,  230000.,  230000.,  225000.,  220000.,
        209000.,  200000.,  200000.,  195000.,  190000.,  180000.,
        180000.,  170000.,  165000.,  165000.,  160000.,  155000.,
        150000.,  150000.,  147500.,  140000.,  135000.,  135000.,
        130000.,  129000.,  124000.,  120000.,  114000.,  110000.,
        110000.,  110000.,  110000.,  107000.,  105000.,  104000.,
        100000.,   93000.,   90000.,   85000.,   80000.,   77000.,
         75000.,   72000.,   70000.,   35000.];

var trace10 = {
  x:sal_us,
  y:sal_us_count,
  mode:"markers",
  marker:{size:5,color:[52,51,50,49,48,47,46,45,44,43,42,41,40,39,38,37,36,35,34,33,32,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1]
  }

};

var layout9 = {
  title: {text:"The salary of ds/ml sp. for the United States"},
  xaxis: {title:{text:"United States's Salary"},
          ticks:'',
          showticklabels:false},
  yaxis: {title:{text:"Avg Salary per year in $"}}
};
var data10 = [trace10];
var config = {responsive: true,scrollZoom: true,displayModeBar: true};
Plotly.newPlot("us_salary",data10,layout9,config);

//------------------------------------Hobby as Coding PIE---------------------------------------------//

var hob = ['Yes', 'No'];
var hob_values = [70.73,29.26];

var trace11 = {
  labels:hob,
  values:hob_values,
  type:'pie',
  textinfo:"label+percent",
  automargin:true
};

var layout10 = {
  margin:"auto",
  title:{text:"Data scientist/ml specialist who code as hobby",autosize:true},
};

var data11 = [trace11];
var config = {responsive:true,scrollZoom: true,displayModeBar: true}

Plotly.newPlot("hobby",data11,layout10,config);



//-------------------------------------------------------------------------//
