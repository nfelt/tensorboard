# Description:
# Summary API for TensorBoard.

licenses(["notice"])  # Apache 2.0

py_library(
    name = "summary",
    srcs = [
        "__init__.py",
        "v1.py",
        "v2.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorboard/plugins/audio:summary",
        "//tensorboard/plugins/custom_scalar:summary",
        "//tensorboard/plugins/histogram:summary",
        "//tensorboard/plugins/image:summary",
        "//tensorboard/plugins/pr_curve:summary",
        "//tensorboard/plugins/scalar:summary",
        "//tensorboard/plugins/text:summary",
    ],
)

package_group(
    name = "summary_nodep_group",
    packages = [
        "//tensorboard/...",
    ],
)

py_library(
    name = "summary_nodep",
    srcs = [
        "__init__.py",
        "v1.py",
        "v2.py",
    ],
    srcs_version = "PY2AND3",
    visibility = [":summary_nodep_group"],
    deps = [
        "//tensorboard/plugins/audio:summary_nodep",
        "//tensorboard/plugins/custom_scalar:summary_nodep",
        "//tensorboard/plugins/histogram:summary_nodep",
        "//tensorboard/plugins/image:summary_nodep",
        "//tensorboard/plugins/pr_curve:summary_nodep",
        "//tensorboard/plugins/scalar:summary_nodep",
        "//tensorboard/plugins/text:summary_nodep",
    ],
)

py_test(
    name = "summary_test",
    size = "small",
    srcs = ["summary_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":summary",
        "//tensorboard:expect_tensorflow_installed",
    ],
)
