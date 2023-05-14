---
title: 'Bug report'
labels: bug
assignees: PunGrumpy
---

on {{ date | date('dddd, MMMM Do') }} at {{ date | date('h:mm a') }}

# Model training failed ⚠️

The model training action failed. Please see the logs for more information. ⚠️

## Logs

<details>
  <summary>Click to expand</summary>

    ```
    ${{ github.event.inputs.logs }}
    ```

</details>
