/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"fmt"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/util/intstr"
)

// FrontendDefaults implements ComponentDefaults for Frontend components
type FrontendDefaults struct {
	*BaseComponentDefaults
}

func NewFrontendDefaults() *FrontendDefaults {
	return &FrontendDefaults{&BaseComponentDefaults{}}
}

func (f *FrontendDefaults) getWorkingDir(context ComponentContext) string {
	switch context.BackendFramework {
	case BackendFrameworkVLLM:
		return "/workspace/components/backends/vllm"
	case BackendFrameworkSGLang:
		return "/workspace/components/backends/sglang"
	case BackendFrameworkTRTLLM:
		return "/workspace/components/backends/trtllm"
	default:
		return "" // signal no working dir default available for this framework
	}
}

func (f *FrontendDefaults) GetBaseContainer(context ComponentContext) (corev1.Container, error) {
	// Frontend doesn't need backend-specific config
	container := f.getCommonContainer(context)

	// Set working directory based on backend framework if available
	if workingDir := f.getWorkingDir(context); workingDir != "" {
		container.WorkingDir = workingDir
	}

	// Set default command and args
	if context.BackendFramework == BackendFrameworkSGLang {
		// For SGLang, we need to clear the namespace first
		container.Command = []string{"sh", "-c"}
		container.Args = []string{
			fmt.Sprintf("python3 -m dynamo.sglang.utils.clear_namespace --namespace %s && python3 -m dynamo.frontend", context.DynamoNamespace),
		}
	} else {
		container.Command = []string{"python3"}
		container.Args = []string{"-m", "dynamo.frontend"}
	}

	// Add HTTP port
	container.Ports = []corev1.ContainerPort{
		{
			Protocol:      corev1.ProtocolTCP,
			Name:          commonconsts.DynamoContainerPortName,
			ContainerPort: int32(commonconsts.DynamoServicePort),
		},
	}

	// Add frontend-specific defaults
	container.LivenessProbe = &corev1.Probe{
		ProbeHandler: corev1.ProbeHandler{
			HTTPGet: &corev1.HTTPGetAction{
				Path: "/health",
				Port: intstr.FromString(commonconsts.DynamoContainerPortName),
			},
		},
		InitialDelaySeconds: 60,
		PeriodSeconds:       60,
		TimeoutSeconds:      30,
		FailureThreshold:    10,
	}

	container.ReadinessProbe = &corev1.Probe{
		ProbeHandler: corev1.ProbeHandler{
			Exec: &corev1.ExecAction{
				Command: []string{
					"/bin/sh",
					"-c",
					"curl -s http://localhost:${DYNAMO_PORT}/health | jq -e \".status == \\\"healthy\\\"\"",
				},
			},
		},
		InitialDelaySeconds: 60,
		PeriodSeconds:       60,
		TimeoutSeconds:      30,
		FailureThreshold:    10,
	}

	container.Resources = corev1.ResourceRequirements{
		Requests: corev1.ResourceList{
			corev1.ResourceCPU:    resource.MustParse("1"),
			corev1.ResourceMemory: resource.MustParse("2Gi"),
		},
		Limits: corev1.ResourceList{
			corev1.ResourceCPU:    resource.MustParse("1"),
			corev1.ResourceMemory: resource.MustParse("2Gi"),
		},
	}

	// Add standard environment variables
	container.Env = append(container.Env, []corev1.EnvVar{
		{
			Name:  commonconsts.EnvDynamoServicePort,
			Value: fmt.Sprintf("%d", commonconsts.DynamoServicePort),
		},
		{
			Name:  "DYNAMO_HTTP_PORT", // TODO: need to reconcile DYNAMO_PORT and DYNAMO_HTTP_PORT
			Value: fmt.Sprintf("%d", commonconsts.DynamoServicePort),
		},
	}...)

	return container, nil
}
