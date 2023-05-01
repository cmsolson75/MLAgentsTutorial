using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Random = UnityEngine.Random;

public class MoveToGoalAgent : Agent
{
    // target position
    [SerializeField] private Transform targetTransform;
    
    // Helpful Visualization For Example
    [SerializeField] private Material winMaterial;
    [SerializeField] private Material loseMaterial;
    [SerializeField] private MeshRenderer floorMeshRenderer;
    

    // Reset agent
    public override void OnEpisodeBegin()
    {
        // transform.localPosition = new Vector3(0, 0.5f, 0);
        transform.localPosition = new Vector3(Random.Range(-4.0f, 4.0f), 0.5f, Random.Range(-4.0f, 4.0f));
        targetTransform.localPosition = new Vector3(Random.Range(-4.3f, 4.3f), 0.5f, Random.Range(-4.3f, 4.3f));

    }
    // This is the input the MLAgent
    public override void CollectObservations(VectorSensor sensor)
    {
        // Telling the MLAgent Where it is
        sensor.AddObservation(transform.position);
        
        // Target Position
        sensor.AddObservation(targetTransform.position);
    }
    
    // Method for giving the Agent actions it can perform
    public override void OnActionReceived(ActionBuffers actions)
    {
        float moveX = actions.ContinuousActions[0];
        float moveZ = actions.ContinuousActions[1];
        // int DiscreteExample = actions.DiscreteActions[0];

        float moveSpeed = 4f;
        
        transform.position += new Vector3(moveX, 0, moveZ) * Time.deltaTime * moveSpeed;
    }
    
    // Method for testing
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        ActionSegment<float> continuousActions = actionsOut.ContinuousActions;
        continuousActions[0] = Input.GetAxisRaw("Horizontal");
        continuousActions[1] = Input.GetAxisRaw("Vertical");
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.TryGetComponent<Goal>(out Goal goal))
        {
            SetReward(1f);
            floorMeshRenderer.material = winMaterial;
            EndEpisode();
        }

        if (other.TryGetComponent<Wall>(out Wall wall))
        {
            SetReward(-1f);
            floorMeshRenderer.material = loseMaterial;
            EndEpisode();
        }
    }
}