import click

from dolos.methods.patch_forensics.evaluate import evaluate_detection, evaluate_localisation


@click.command()
@click.option("-s", "supervision", type=click.Choice(["full", "weak"]), required=True)
@click.option("-t", "train_config_name", required=True)
@click.option("-d", "dataset_name", required=True)
@click.option("-v", "to_visualize", is_flag=True, default=False)
def main(supervision, train_config_name, dataset_name, to_visualize=False):
    method_name = "xception-attention"
    print(supervision, train_config_name)
    if supervision == "weak" and dataset_name in {
        "repaint-clean",
        "repaint-p2-celebahq-clean",
    }:
        evaluate_detection(method_name, supervision, train_config_name, dataset_name, to_visualize)
    evaluate_localisation(method_name, supervision, train_config_name, dataset_name, to_visualize)
    print()


if __name__ == "__main__":
    main()  # type: ignore
