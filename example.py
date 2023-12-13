from flask import Flask, redirect, url_for, render_template

app = Flask(__name__)


@app.route("/")
def home():
    # return "Hello! this is the mail page <h1>HELLO</h1>"
    # return render_template("index.html", content=["tim", "joe", "bill"])
    return render_template("index.html")


@app.route("/page_calibration")
def page_calibration():
    return render_template("page_calibration.html")


@app.route("/page_chars_of_camera")
def page_chars_of_camera():
    return render_template("page_chars_of_camera.html")


@app.route("/page_aruko_markers")
def page_aruko_markers():
    return render_template("page_aruko_markers.html")


@app.route("/page_def_of_color")
def page_def_of_color():
    return render_template("page_def_of_color.html")


@app.route("/page_s_kontur")
def page_s_kontur():
    return render_template("page_s_kontur.html")


@app.route("/page_bwhite")
def page_bwhite():
    return render_template("page_bwhite.html")


@app.route("/page_expo")
def page_expo():
    return render_template("page_expo.html")


@app.route("/page_round_kontur")
def page_round_kontur():
    return render_template("page_round_kontur.html")


@app.route("/page_where_object")
def page_where_object():
    return render_template("page_where_object.html")


@app.route("/page_figures")
def page_figures():
    return render_template("page_figures.html")


@app.route("/page_lines")
def page_lines():
    return render_template("page_lines.html")


@app.route("/page_recog_ai")
def page_recog_ai():
    return render_template("page_recog_ai.html")


@app.route("/page_fpv_from_face")
def page_fpv_from_face():
    return render_template("page_fpv_from_face.html")

# examples
@app.route("/test")
def test():
    return render_template("new.html")


@app.route("/<name>")
def user(name):
    return f"Hello {name}!"


@app.route("/admin")
def admin():
    # return redirect(url_for("home"))
    return redirect(url_for("user", name="Admin!"))


if __name__ == "__main__":
    app.run(debug=True)
