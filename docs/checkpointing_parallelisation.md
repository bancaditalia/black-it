# Saving and loading

For each batch execution of the calibration, a checkpoint can be created (*save*) so that it can be later restored (*load*).

The functions responsible for these features are the method `create_checkpoint(file_name)` and the class method `restore_from_checkpoint(checkpoint_path, model)` of the `Calibrator` class.

The checkpoint is created automatically at each batch execution, if a saving folder is provided. To do this, when a `Calibrator` object is instantiated, the `saving_folder` field must be specified as a string which contains the path where the checkpoint must be saved.

*Remark*: there's no need to call the save method, once the saving folder is specified.

To restore the checkpoint it suffices to call the load class method with the path name and, optionally, the model which was being calibrated. Once restored, the calibration starts from where it was interrupted.

An example of save and load is as follows:
```python
# initialize a Calibrator object
    cal = Calibrator(
        samplers=[
            random_sampler,
            halton_sampler,
        ],
        real_data=real_data,
        model=model,
        parameters_bounds=bounds,
        parameters_precision=bounds_step,
        ensemble_size=2,
        loss_function=loss,
        saving_folder="saving_folder",
        n_jobs=1,
    )

    _, _ = cal.calibrate(2)

    cal_restored = Calibrator.restore_from_checkpoint("saving_folder", model=model)
```

This code has been extracted from the following example: **`tests/test_calibrator_restore_from_checkpoint.py`**.

*Remark*: the saving folder where the checkpoint is saved can also be used as a parameter for the plotting functions, as shown in the [plotting tutorial](overview_of_plotting_functions.ipynb), to produce plots quickly.

# Parallelisation

Since calibrating an agent-based model can be very intensive, by default the model simulation is parallelised. The number of parallel processes equals the number of cores in the computer. This number can be changed by specifying the optional parameter `n_jobs` in the constructor of the calibrator.
