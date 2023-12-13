(".polzunok-2").slider({
    min: 0,
    max: 500,
    value: 200,
    range: "min",
    animate: "fast",
    slide : function(event, ui) {    
        $(".polzunok-2 span").html("<b>&lt;</b>" + ui.value + "<b>&gt;</b>");        
    }    
});
(".polzunok-2 span").html("<b>&lt;</b>" + $(".polzunok-2").slider("value") + "<b>&gt;</b>");