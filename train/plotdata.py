import pandas as pd
import numpy as np


def RGB_to_Hex(rgb):
    RGB = rgb.split(',')  # 将RGB格式划分开来
    color = '\'#'
    for i in RGB:
        num = int(i)
        # 将R、G、B分别转化为16进制拼接转换并大写  hex() 函数用于将10进制整数转换成16进制，以字符串形式表示
        color += str(hex(num))[-2:].replace('x', '0').upper()
    return color + '\''


col1name = '时间'
d1 = pd.read_excel('data.xlsx')
col1 = np.unique(d1.iloc[:, 0])

colnames = list(d1.columns)

col2 = d1[d1.iloc[:, 0] == col1[0]].sort_values(by=colnames[-1], ascending=True).iloc[:, 1].tolist()
text = '''
<!DOCTYPE html><html><head><meta charset="UTF-8"><title></title>
<script src="C:/Users\Administrator\Downloads\ECI数据库\结果统计\plotjscode\g2\dist\g2.min.js"></script>
<script src="C:/Users\Administrator\Downloads\ECI数据库\结果统计\plotjscode\data-set\dist\data-set.js"></script>
</head><body><div id="container"></div><script>
const data = [
'''

for i in range(len(col1)):
    temp = d1[d1.iloc[:, 0] == col1[i]].sort_values(by=colnames[-1], ascending=True)
    col = temp.iloc[:, 1].values
    text += '{State:\'' + str(col1[i]) + '\','
    for j in range(len(col)):
        text += '\'' + col[j] + colnames[2] + '\':' + str(temp.iloc[j, 2]) + ','
        text += '\'' + col[j] + colnames[3] + '\':' + str(temp.iloc[j, 3]) + ','
    text += '},'
text += '];const ages = ['
for j in range(len(col2)):
    text += '\'' + col2[j] + colnames[2] + '\','
    text += '\'' + col2[j] + colnames[3] + '\','

text += '];const dv = new DataSet.DataView();dv.source(data).transform({type: \'fold\',fields: ages,key: \'age\',value: \'' \
        'population\',retains: [\'State\'],}).transform({type: \'map\',callback: (obj) => {const key = obj.age;let type;'
text += 'if ('
for j in range(len(col2)):
    text += 'key===\'' + col2[j] + colnames[2] + '\'||'
text = text[:-2]
text += ' ){type = \'a\';}else {type = \'b\';}obj.type = type;return obj;},}); '

color_range = np.arange(0, 255, len(col2))[::-1].tolist()
colorx = np.zeros((len(col2), 2), dtype=object)
for i in range(len(col2)):
    colorx[i, 0] = RGB_to_Hex('0,0,' + str(color_range[i]))
    colorx[i, 1] = RGB_to_Hex('0,' + str(color_range[i]) + ',0')

text += 'const colorMap = {'
for j in range(len(col2)):
    text += '\'' + col2[j] + colnames[2] + '\':' + colorx[j, 0] + ','
    text += '\'' + col2[j] + colnames[3] + '\':' + colorx[j, 1] + ','

text += '};'
text1 = '''

const chart = new G2.Chart({
container: 'container',
autoFit: true,
height: 500,
});

chart.data(dv.rows);

chart.scale({
population: {
tickInterval: 50000000,
},
});

chart.axis('population', {
label: {
formatter: (val) => {
return +val / 10000 + '万';
},
},
});
chart.legend({
position: 'right-bottom',
});

chart.tooltip({
showMarkers: false,
shared: true,
});

chart
.interval()
.position('State*population').color('age', (age) => {
return colorMap[age];
})
.tooltip('age*population', (age, population) => {
return {
name: age,
value: population,
};
})
.adjust([
{
type: 'dodge',
dodgeBy: 'type', // 按照 type 字段进行分组
marginRatio: 0, // 分组中各个柱子之间不留空隙
},
{
type: 'stack',
},
]);

chart.interaction('active-region');

chart.render();
</script>
</body>
</html>
'''
text += text1
file_handle = open('plot1.html', mode='w', encoding='utf-8')
file_handle.write(text)

file_handle.close()
print(1)
