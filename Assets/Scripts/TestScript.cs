using System.Collections;
using System.Collections.Generic;
using Unity.Barracuda;
using UnityEngine;

public class TestScript : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        Debug.Log("Hello World!!!!");
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKey(KeyCode.A))
        {
            float speed = -5f;
            transform.position += new Vector3(speed, 0, 0) * Time.deltaTime;
        }

        if (Input.GetKeyDown(KeyCode.F))
        {
            Debug.Log("Pew Pew");
        }
    }
}
