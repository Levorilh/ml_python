let app = {
    hasModel: false,
    modelId : null,
};

const MONUMENTS = ["Tour eiffel", "Arc de Triomphe", "Notre dame"];
$("#insert").on("change", function () {
    const [file] = this.files;
    const $preview = $("#preview");
    if (file) {
        $preview.attr("src", URL.createObjectURL(file));
        $('#test').removeClass('d-none');
        $preview.removeClass('d-none');

    } else {

        console.log('rien ne va');

    }
});

$("#test").on("click", function () {
    if (app.hasModel) {
        $.ajax({
            method: "POST",
            url: '/predict',
            success: function (html) {
                $("#outputArea").html(html);
            }
        });
    }
    else{
        window.alert("Vous devez d'abord charger un modèle");
    }

    // let result_class = Math.round(Math.random()*MONUMENTS.length);
    // let result_precision = (Math.random()*100).toPrecision(4);
});

$(".modelLoader").on("click", function () {
    console.log("touch me");
    let modelId = $(this).data("id");
    $.ajax({
        url: "/setModel",
        method: "POST",
        data: {
            "id": modelId,
        },
        success: function (response, status) {
            console.log(response, status);
            if (status === "success") {
                window.alert("Le modèle " + modelId + " a été chargé");
                app.hasModel = true;
                app.modelId = modelId;
                // console.log("Le modèle " + modelId + " a été chargé")
            }
        },
    })
});