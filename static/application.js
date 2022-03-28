new Chart(document.getElementById("myChart2"), {
    type: 'line',
    data: {
    lables1: labels1,
    datasets: [{
        data: values1,
        label: "Positive 👍",
        borderColor: "#3e95cd",
        fill: false
        }, {
        data: values2,
        label: "Negative 👎",
        borderColor: "#8e5ea2",
        fill: false
        }
    ]
    },
    options: {
    title: {
        display: true,
        text: 'Line Grphs'
    },
    hover: {
    mode: 'index',
    intersect: true
    },
    }
});